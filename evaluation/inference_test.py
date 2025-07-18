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
from T2I_Eval.genaibench_generation import sample
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

    gpt_text = ""

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
    with open("/data/tempdata_val/000000.jsonl", "r", encoding="utf-8") as f:
        sources = json.loads(f.readline())
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
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': False}
        cur_len = input_ids.shape[1]+256*3
        model_kwargs = {'attention_mask': attention_mask, 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device="cuda:0")

        pred_tokens = []
        pred_logits = []
        image_insert_pos = [271 * i for i in range(6)]
        image_index = 0
        total_steps = 1626
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
        for i in tqdm(range(total_steps)):
            # model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
            seq_len = inputs_embeds.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(
                0)  # shape: [1, seq_len]
            # print("attention mask:",attention_mask)

            attention_mask = new_input_ids.ne(tokenizer.pad_token_id)

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
            # print(outputs.keys())
            next_embed = outputs['last_hidden_state'][:, -1:, :]  # [1, 1, vocab_size]
            inputs_embeds = torch.cat((inputs_embeds,next_embed), dim=1)
            print("next_embeds:", next_embed.shape)
            indices_arhead = []
            for i_head in range(num_codebooks):
                ar_next_embed = vqllm.ar_head(
                    inputs_embeds=next_embed,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                )
                next_token_logits = vqllm.ar_head.linear_head(ar_next_embed[0])
                # print("next_token_logits:", next_token_logits.shape)
                # pred_logits.append(next_token_logits)
                # if cfg_scale > 1:
                #     cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                #     cfg_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                #     half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                #     # pred_tokens.append(half_next_token)
                #     next_token = torch.cat([half_next_token, half_next_token])  # [bz,1]
                # else:
                next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                    # pred_tokens.append(next_token)
                indices_arhead.append(next_token)
                if i_head < num_codebooks - 1:
                    predicted_embed = vqllm.ar_head.codebooks[i_head](next_token)
                    next_embed = torch.cat([next_embed, predicted_embed], dim=1)
            # print("next_embeds:", next_embed.shape)
            pred_logits.append(next_token_logits)
            pred_tokens.append(torch.cat(indices_arhead, dim=1))  # [numcodebook,bz*2]
            print("len:",len(pred_tokens))
            print("pred_tokens:", pred_tokens[0].shape)
            # input_multi_ids = torch.stack(pred_tokens, dim=-1)
            # fake_id = torch.zeros_like(input_ids[:, :1])
            # input_ids = torch.cat([input_ids, fake_id], dim=-1)  # add fake id for cache
            model_kwargs["cache_position"] = torch.arange(inputs_embeds.shape[1], device="cuda:0")
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )
            in_image_range = any(p <= i < p + 256 for p in image_insert_pos)
            if in_image_range:
                new_input_ids = torch.cat([new_input_ids, torch.tensor([[IMAGE_TOKEN_INDEX]]).to("cuda")], dim=-1)
            else:
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
        generated_ids = torch.cat(pred_tokens, dim=0)  # [T]
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
        print("小于 256000 的数量为：", count)
        # # 确保找到6个<boi>标记
        # assert len(boi_pos) == 6, f"Expected 6 <boi> tokens, found {len(boi_pos)}"
        boi_pos = np.arange(6) * 271
        img_logits = []
        for pos in boi_pos:
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
        image_ce_loss = criterion(img_logits.reshape(-1,4096), gt_img_tokens.view(-1))
        print("📉 Average Image CrossEntropy Loss:", image_ce_loss.item())

        # ====== 解码图像 & 可视化对比 ======
        vq_token_lists = []
        for i in range(len(boi_pos)):
            start = boi_pos[i] + 1
            end = start + 256
            vq_token = generated_ids[start:end,:].permute(1,0)
            vq_token_lists.append(vq_token)

        # pic_ori = os.path.basename(pic_path)
        # pic_num = int(pic_ori.split(".")[0])
        # pic_dir = os.path.dirname(pic_path)
        # pred_vqcodes = torch.stack(vq_token_lists, dim=0).to("cuda")  # [6, 256]
        # pred_vqcodes = pred_vqcodes - len(tokenizer)
        # pred_vqcodes = torch.clamp(pred_vqcodes, 0, 8191)
        future_vqcodes = torch.stack(future_vqcodes, dim=0).to("cuda")
        for i, vq_token in enumerate(vq_token_lists):
            new_gen_ids = vq_token.unsqueeze(0).to('cuda')
            print("new_gen_ids:", new_gen_ids.shape)
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

            # 将 torch.Tensor 转换为 PIL.Image
            rec_img_pil = to_pil(rec_img.squeeze(0).cpu().clamp(0, 1))  # [C, H, W]
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


