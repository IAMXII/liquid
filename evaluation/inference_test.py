import torch
import numpy as np
import argparse
import time
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os
from tqdm import tqdm
from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import conv_templates
from threading import Thread
from torch.nn import functional as F

from torchvision import transforms
PILtransform = transforms.ToPILImage()
from torch.nn import CrossEntropyLoss
import torchvision.transforms as T
from PIL import Image
import json
from liquid.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                              DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from torch.utils.data import Dataset
from liquid.train.llava_trainer import LLaVATrainer

from liquid import conversation as conversation_lib
from liquid.model import *
from liquid.mm_utils import tokenizer_image_token
from liquid.model.language_model.mini_gemini_gemma import MiniGeminiGemmaForCausalLM
from config import Args
from unitok import UniTok

def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k

        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    # import pdb;pdb.set_trace()
    return logits


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def format_wp(wp):  # 格式化为 "[x,y]"
    return f"[{wp[0]:.2f},{wp[1]:.2f}]"


def build_vqa_pair_with_vqcode(tokenizer, sources):
    known_wps = sources["known_waypoints"]
    future_wps = sources["future_waypoints"]

    def make_vqtext(wp):
        return "<boi><eoi>" + format_wp(wp)

    human_text = "We already know three frames and their waypoints: " + \
                 ", ".join([make_vqtext(wp) for wp in known_wps]) + \
                 ", now predict the next frames, their waypoints are " + \
                 ", ".join([format_wp(wp) for wp in future_wps])

    gpt_text = "<boi>"

    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], human_text)
    conv.append_message(conv.roles[1], gpt_text)
    prompt = conv.get_prompt()

    input_ids = \
        tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).input_ids[0]
    instruction_len = len(
        tokenizer(human_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).input_ids[0])
    instruction_len += 1 * 3 + 4

    # 替换每对 <boi><eoi> 中插入 IMAGE_TOKEN_INDEX
    def insert_image_token_placeholders(input_ids):
        output = []
        i = 0
        while i < len(input_ids):
            token = input_ids[i].item()
            token_str = tokenizer.convert_ids_to_tokens(token)
            if token_str == "<boi>":
                output.append(token)  # <boi>
                i += 1
                output.append(IMAGE_TOKEN_INDEX)  # 插入占位
                # 跳过原来的 <eoi>
                while i < len(input_ids):
                    token = input_ids[i].item()
                    if tokenizer.convert_ids_to_tokens(token) == "<eoi>":
                        output.append(token)
                        i += 1
                        break
                    i += 1
            else:
                output.append(token)
                i += 1
        # eos_id = tokenizer.convert_tokens_to_ids("<eos>")
        output.append(1)  # <eos>
        return torch.tensor(output, dtype=torch.long)

    input_ids = insert_image_token_placeholders(input_ids)

    labels = input_ids.clone()
    labels[:instruction_len] = IGNORE_INDEX

    return input_ids.unsqueeze(0).to("cuda"), labels.unsqueeze(0).to("cuda")


def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    Width, Height = ori_image.size
    factor = min(Width, Height) / min(tgt_width, tgt_height)
    input_image = ori_image.resize((int(Width / factor), int(Height / factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size  # Get dimensions

    left = (resize_width - tgt_width) // 2
    top = (resize_height - tgt_height) // 2
    right = (resize_width + tgt_width) // 2
    bottom = (resize_height + tgt_height) // 2
    # Crop the center of the image
    input_image = input_image.crop((left, top, right, bottom))
    return input_image


# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_uni']
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
#             continue
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#
#     if 'lm_head' in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)


def main(args):
    num_codebooks = 8
    temperature = args.temperature
    guidance_scale = args.cfg
    top_K = args.TopK
    top_P = args.TopP
    image_save_pth = args.save_path
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth)

    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    ori_vocab_size = len(tokenizer)

    # vqllm = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     attn_implementation='flash_attention_2',
    #     torch_dtype=torch.bfloat16,
    #     load_in_8bit=args.load_8bit,
    # )

    vqllm = MiniGeminiGemmaForCausalLM.from_pretrained(  ### liuwei
        # model = MiniGeminiLlamaForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
    )

    if not args.load_8bit:
        vqllm = vqllm.to('cuda')

    # vqgan_cfg_path = "model/vqgan_imagenet_f16_1024/configs/model.yaml"
    # vqgan_ckpt_path = "model/vqgan_imagenet_f16_1024/ckpts/last.ckpt"
    # image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0")
    unitok_path = "../unitok/unitok_tokenizer/unitok_tokenizer.pth"
    print('loading vq model ...')
    ckpt = torch.load(unitok_path, map_location='cpu')
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])
    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    vq_model.to('cuda')
    vq_model.eval()
    # ==== 加载数据 ====
    with open("/data/tempdata/000000.jsonl", "r", encoding="utf-8") as f:
        # sources = json.loads(f.readline())
        for i, line in enumerate(f):
            if i == 999:  # 第1000行的索引是999
                sources = json.loads(line)
                break
        future_vqcodes = [torch.tensor(json.loads(s)) for s in sources["future_vqcodes"]]
        gt_img_tokens = torch.cat(future_vqcodes, dim=0).to("cuda")  # [6*256]

    # pic_path = sources["pic_path"]
    # input_ids, attention_mask = build_vqa_inference_input(tokenizer, sources)

    input_ids, labels = build_vqa_pair_with_vqcode(tokenizer, sources)
    known_vqcodes = [torch.tensor(json.loads(s)) for s in sources["known_vqcodes"]]
    future_vqcodes = [torch.tensor(json.loads(s)) for s in sources["future_vqcodes"]]
    # print(known_vqcodes[0])
    # vqcode = known_vqcodes + future_vqcodes
    vqcode = known_vqcodes
    # vqcode = torch.tensor(vqcode)
    vqcode = torch.stack(vqcode, dim=0)
    vqcode = vqcode.unsqueeze(0)
    vqcode = vqcode.to("cuda")
    # processed_instances.append(dict(
    #     input_ids=input_ids,
    #     labels=labels,
    #     image=vqcode,
    #     data_type=5
    # ))
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    input_ids_flat = input_ids[0]

    # 新 tensor 列表
    new_ids = []

    for token in input_ids_flat:
        if token.item() == IMAGE_TOKEN_INDEX:
            new_ids.extend([IMAGE_TOKEN_INDEX] * 256)
        else:
            new_ids.append(token.item())

    # 转回 tensor，加上 batch 维度
    new_input_ids = torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
    with torch.no_grad():
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        cur_len = input_ids.shape[1] + 256 * 3
        model_kwargs = {'attention_mask': attention_mask, 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device="cuda:0")

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            data_types,
            additional_image_labels,
            additional_image_indexs
        ) = vqllm.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=vqcode,
            images_aux=None,
            data_types=[5]
        )
        pred_tokens = []
        pred_logits = []
        image_insert_pos = [269 * i for i in range(6)]
        total_steps = 1626

        is_last_image_embed = False  # 用于标记前一步是否是图像embedding

        for i in tqdm(range(total_steps)):
            in_image_range = any(p <= i < p + 256 for p in image_insert_pos)
            # 决定 inputs_embeds 的裁剪范围
            # if is_last_image_embed:
            #     # 查找当前图像的插入位置，并裁剪到其开始前
            #     current_img_idx = max([j for j, pos in enumerate(image_insert_pos) if pos <= i])
            #     img_start_pos = image_insert_pos[current_img_idx]
            #     input_chunk = inputs_embeds[:, img_start_pos:]
            # else:
            #     input_chunk = inputs_embeds

            seq_len = inputs_embeds.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
            attention_mask = new_input_ids.ne(tokenizer.pad_token_id)
            # len_input = input_chunk.size(1)
            # attention_mask = attention_mask[:, -seq_len:]
            outputs = vqllm.T2I_forward_withcache(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                input_multi_ids=None,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_embed = outputs['last_hidden_state'][:, -1:, :]  # 下一个 token embedding

            # next_embed_t = next_embed
            indices_arhead = []
            # is_last_image_embed = True  # 默认下一步是图像
            if in_image_range:

                for i_head in range(num_codebooks):
                    ar_next_embed = vqllm.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    next_token_logits = vqllm.ar_head.linear_head(ar_next_embed[0])
                    # print("next_token_logits", next_token_logits)
                    next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                    indices_arhead.append(next_token)

                    # 若不是最后一层 head，则准备下一个嵌入
                    if i_head < num_codebooks - 1:
                        predicted_embed = vqllm.ar_head.codebooks[i_head](next_token)
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)

                pred_logits.append(next_token_logits)
                pred_tokens.append(torch.cat(indices_arhead, dim=1))
                next_token = torch.stack(pred_tokens, dim=-1)
                next_embed = vqllm.get_model().multi_embedder(next_token)
            # fake id for cache & extend full embedding序列
            # fake_id = torch.zeros_like(next_embed).to(next_embed.device)

            inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
            # 更新 cache 与输入
            model_kwargs["cache_position"] = torch.arange(inputs_embeds.shape[1], device="cuda:0")
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

            # 判断当前位置是否是插图区域，用于决定 new_input_ids 拼接什么 token



            if in_image_range:
                new_input_ids = torch.cat([new_input_ids, torch.tensor([[IMAGE_TOKEN_INDEX]]).to("cuda")], dim=-1)
            else:
                if i in [x - 1 for x in image_insert_pos]:
                    next_token = torch.tensor([[7]]).to("cuda")  # <boi>
                    is_last_image_embed = True  # boi 是文本 token
                elif i in [x + 256 for x in image_insert_pos]:
                    next_token = torch.tensor([[8]]).to("cuda")  # <eoi>
                    is_last_image_embed = False
                else:
                    next_token = torch.tensor([[0]]).to("cuda")
                new_input_ids = torch.cat([new_input_ids, next_token], dim=-1)


            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

        # generated_ids = vqllm.generate(
        #     input_ids=input_ids,
        #     max_new_tokens=1626,
        #     do_sample=False,  # 如果你原来是用 sample() 采样
        #     **model_kwargs  # 包括 attention_mask 或 past_key_values
        #     )
        # print("gen:", generated_ids)
        # print("gen shape:", generated_ids.shape)
        # ====== 生成后处理 ======
        # print("pred_logits:",len(pred_logits))
        # print("pred_logits:", pred_logits[0].shape)
        # generated_ids = torch.cat(pred_tokens, dim=0)  # [T]
        generated_ids = torch.stack(pred_tokens, dim=-1)
        print("generated_ids:", generated_ids.shape)
        full_logits = torch.cat(pred_logits, dim=0)  # [1, T, vocab_size]
        full_logits = full_logits.permute(1, 0, 2)  # shape: [X, B, Y]
        # full_logits = full_logits.reshape(-1, full_logits.size(-1))  # shape: [X*B, Y]
        # ====== 提取图像 token 段 ======
        # boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        # # print("gen:", generated_ids)
        # print("boi:", boi_token_id)
        # boi_pos = (generated_ids == boi_token_id).nonzero(as_tuple=True)[0]
        count = (generated_ids < 256000).sum().item()
        # print("小于 256000 的数量为：", count)
        # # 确保找到6个<boi>标记
        # assert len(boi_pos) == 6, f"Expected 6 <boi> tokens, found {len(boi_pos)}"
        # boi_pos = np.arange(6) * 271+1
        img_logits = []
        pos_logits = np.arange(6)*256
        for pos in pos_logits:
            start = pos
            end = start + 256
            img_logits.append(full_logits[:, start:end])  # 每张256个 token

        img_logits = torch.stack(img_logits, dim=0)  # [6, 256, vocab]
        criterion = CrossEntropyLoss()

        # 确保维度匹配
        # assert img_logits.shape[0] * img_logits.shape[1] == gt_img_tokens.numel(), \
        #     f"Shape mismatch: img_logits {img_logits.shape} vs gt_img_tokens {gt_img_tokens.shape}"
        # print("img shape: ", img_logits.shape)
        # image_ce_loss = criterion(img_logits.reshape(-1, 264192), gt_img_tokens.view(-1))
        # image_ce_loss = criterion(img_logits.reshape(-1, 4096), gt_img_tokens.view(-1))
        # print("📉 Average Image CrossEntropy Loss:", image_ce_loss.item())
        image_losses = []
        img_loss_l = img_logits.reshape(-1, 4096)

        for i in range(6):
            start = i * 256
            end = (i + 1) * 256

            logits_i = img_loss_l[start:end, :]  # [256, 8192]
            targets_i = gt_img_tokens.view(-1)[start:end]  # [256]

            loss_i = criterion(logits_i, targets_i)
            image_losses.append(loss_i.item())
            print(f"📉 Image {i} CrossEntropy Loss:", loss_i.item())

        avg_loss = sum(image_losses) / len(image_losses)
        print("📉 Average Image CrossEntropy Loss:", avg_loss)

        # ====== 解码图像 & 可视化对比 ======
        vq_token_lists = []
        for i in range(len(image_insert_pos)):
            start = pos_logits[i]
            end = start + 256
            vq_token = generated_ids[:, :, start:end]
            print("vq_token:", vq_token.shape)
            vq_token_lists.append(vq_token)

        # pic_ori = os.path.basename(pic_path)
        # pic_num = int(pic_ori.split(".")[0])
        # pic_dir = os.path.dirname(pic_path)
        pred_vqcodes = torch.stack(vq_token_lists, dim=0).to("cuda")  # [6, 256]
        # pred_vqcodes = pred_vqcodes - len(tokenizer)
        pred_vqcodes = torch.clamp(pred_vqcodes, 0, 4095)
        future_vqcodes = torch.stack(future_vqcodes, dim=0).to("cuda")
        # print("equal:",vq_token_lists[0]==vq_token_lists[1])
        for i, vq_token in enumerate(pred_vqcodes):
            new_gen_ids = vq_token.to('cuda')
            # print("new_gen_ids:", new_gen_ids)
            rec_img = vq_model.idx_to_img(new_gen_ids)
            # rec_img = image_tokenizer.pil_from_img_toks(vq_token, height=16, width=16)
            ori_code = future_vqcodes[i].unsqueeze(0).to('cuda')
            ori_img = vq_model.idx_to_img(ori_code)
            # k = pic_num + i
            # ori_path = os.path.join(pic_dir, f"{k:05d}.jpg")
            # if not os.path.exists(ori_path):
            #     print(f"⚠️ 原图不存在: {ori_path}")
            #     continue

            # ori_img = Image.open(ori_path).convert("RGB")
            # ori_img = center_crop_image(ori_img, tgt_width=256, tgt_height=256)

            to_pil = T.ToPILImage()

            # 将 torch.Tensor 转换为 PIL.
            rec_img_pil = PILtransform(rec_img.squeeze(0).add(1).mul_(0.5).clamp_(0, 1))
            # rec_img_pil = to_pil(rec_img.squeeze(0).cpu().clamp(0, 1))  # [C, H, W]
            ori_img_pil = to_pil(ori_img.squeeze(0).cpu().clamp(0, 1))  # 同上

            # 拼接并保存图像
            w, h = ori_img_pil.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(ori_img_pil, (0, 0))
            combined.paste(rec_img_pil, (w, 0))
            combined.save(f"{image_save_pth}/compare_{i}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='Junfeng5/Liquid_V1_7B',
                        help='model path, default to huggingface repo id')
    parser.add_argument('--save_path', type=str, default='samples/t2i', help='save path')
    parser.add_argument('--prompt', type=str, help='input text prompt')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit to save memory')
    parser.add_argument('--cfg', type=float, default=5.0, help='Classifier-Free Guidance scale')
    parser.add_argument('--TopP', type=float, default=0.96, help='Top P, max=1.0')
    parser.add_argument('--TopK', type=int, default=4096, help='Top K, max=264192')
    parser.add_argument('--temperature', type=float, default=0.99, help='sampling temperature, max=1.0')

    args = parser.parse_args()
    main(args)
