import torch
import numpy as np
import argparse
import time
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os

from torchvision import transforms
from tqdm import tqdm
from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import conv_templates
from threading import Thread
from T2I_Eval.genaibench_generation import sample
from torch.nn import CrossEntropyLoss
import json
from torch.nn import functional as F

import torch
import torch.nn.functional as F


def sample_lw(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits: bool = True):
    """
    从 logits 中采样 token。

    参数：
        logits (Tensor): [batch_size, 1, vocab_size]
        temperature (float): 控制分布的平滑程度，越大越随机，越小越确定性
        top_k (int): 保留前 k 个概率最大的 token（top-k 采样）
        top_p (float): nucleus sampling，保留累计概率 >= top_p 的最小集合
        sample_logits (bool): 是否采样。如果为 False，则使用贪婪解码（argmax）

    返回：
        next_token (Tensor): 采样的 token id，形状为 [batch_size, 1]
        probs (Tensor): 当前 token 分布的 softmax 概率
    """
    logits = logits[:, -1, :]
    print(logits)
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k 筛选
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    # Top-p 筛选（Nucleus Sampling）
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 过滤掉 cumulative_probs > top_p 的 token
        sorted_mask = cumulative_probs > top_p
        # 将第一个超过top_p的token后面的全部置为 -inf
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0

        # 把被 mask 掉的位置设置为 -inf
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

        # scatter 回原始顺序
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]

    if sample_logits:
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

    return next_token, probs


def format_wp(wp):  # 格式化为 "[x,y]"
    return f"[{wp[0]:.2f},{wp[1]:.2f}]"


def build_vqa_inference_input(tokenizer, sources):
    tokenizer_len = len(tokenizer)

    def load_vqcode(s):
        return torch.tensor(json.loads(s)) + tokenizer_len

    known_vqcodes = [load_vqcode(s) for s in sources["known_vqcodes"]]
    future_vqcodes = [load_vqcode(s) for s in sources["future_vqcodes"]]
    known_wps = sources["known_waypoints"]
    future_wps = sources["future_waypoints"]

    def make_vqtext(wp):
        return "<boi><eoi>" + format_wp(wp)

    human_text = "We already know three frames and their waypoints: " + ", ".join(
        [make_vqtext(wp) for wp in known_wps]
    ) + ", now predict the next frames, their waypoints are " + ", ".join([format_wp(wp) for wp in future_wps])

    from liquid import conversation as conversation_lib
    conv = conversation_lib.conv_templates["gemma"].copy()
    conv.append_message(conv.roles[0], human_text)
    conv.append_message(conv.roles[1], "")  # 模型生成
    prompt = conv.get_prompt()
    # prompt += " "
    # print(prompt)
    # 原始 input_ids 和 attention_mask（还不含图像 token）
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = encoded["input_ids"][0].tolist()  # 转为 list 方便插入
    attention_mask = encoded["attention_mask"][0].tolist()

    # 插入 VQ token 的函数（返回更新后的 input_ids 和 attention_mask）
    def insert_vqcodes_with_mask(input_ids, attention_mask, vqcode_list):
        output_ids = []
        output_mask = []
        i = 0
        vq_iter = iter(vqcode_list)
        while i < len(input_ids):
            token = input_ids[i]
            token_str = tokenizer.convert_ids_to_tokens(token)
            if token_str == "<boi>":
                output_ids.append(token)
                output_mask.append(0)
                i += 1

                vq = next(vq_iter)  # tensor: [8*256]
                vq_list = vq.tolist()
                output_ids.extend(vq_list)
                output_mask.extend([1] * len(vq_list))

                # 找到 <eoi>
                while i < len(input_ids):
                    if tokenizer.convert_ids_to_tokens(input_ids[i]) == "<eoi>":
                        output_ids.append(input_ids[i])
                        output_mask.append(0)
                        i += 1
                        break
                    i += 1
            else:
                output_ids.append(token)
                output_mask.append(attention_mask[i])
                i += 1
        return torch.tensor(output_ids, dtype=torch.long), torch.tensor(output_mask, dtype=torch.long)

    # 插入图像 token，同时生成新的 attention_mask
    all_vqcodes = known_vqcodes + future_vqcodes
    input_ids, attention_mask = insert_vqcodes_with_mask(input_ids, attention_mask, all_vqcodes)

    return input_ids.unsqueeze(0).to("cuda"), attention_mask.unsqueeze(0).to("cuda")  # shape: [1, T]


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


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_uni']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# def main(args):
#     temperature = args.temperature
#     guidance_scale = args.cfg
#     top_K = args.TopK
#     top_P = args.TopP
#     image_save_pth = args.save_path
#     if not os.path.exists(image_save_pth):
#         os.makedirs(image_save_pth)

#     assert temperature <= 1.0
#     assert top_K <= 264192
#     assert top_P <= 1.0

#     model_id = args.model_path
#     tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
#     ori_vocabe_size = len(tokenizer)
#     # from transformers import AutoConfig

#     # config = AutoConfig.from_pretrained(model_id)
#     # config.rope_scaling = {"type": "linear", "factor": 8.0}  # 提高 RoPE 支持的长度
#     vqllm = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         attn_implementation='flash_attention_2',
#         # attn_implementation='eager',
#         torch_dtype=torch.bfloat16,
#         load_in_8bit=args.load_8bit,
#         # config=config,
#     )
#     # from peft import LoraConfig, get_peft_model
#     # lora_config = LoraConfig(
#     #     r=64,
#     #     lora_alpha=16,
#     #     target_modules=find_all_linear_names(vqllm),
#     #     lora_dropout=0.05,
#     #     bias="none",
#     #     task_type="CAUSAL_LM",
#     # )
#     if not args.load_8bit:
#         vqllm = vqllm.to('cuda')
#     # print(vqllm.config.max_position_embeddings)  # 如果是 98 就是问题

#     # vqllm = get_peft_model(vqllm, lora_config)
#     vqgan_cfg_path = "model/vqgan_imagenet_f16_1024/configs/model.yaml"
#     vqgan_ckpt_path = "model/vqgan_imagenet_f16_1024/ckpts/last.ckpt"
#     image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0", )
#     with open("/data/tempdata_val/000000.jsonl", "r", encoding="utf-8") as f:
#         first_line = f.readline()
#         sources = json.loads(first_line)
#         future_vqcodes = [torch.tensor(json.loads(s)) for s in sources["future_vqcodes"]]  # 每张是 [8*256]
#         gt_img_tokens = torch.cat(future_vqcodes, dim=0).to("cuda")  # shape: [N]
#     # print(sources)
#     pic_path = sources["pic_path"]
#     input_ids, attention_mask = build_vqa_inference_input(tokenizer, sources)
#     # print("input_ids device:",input_ids.device)
#     # print("atten device:",attention_mask.device)
#     # text_inputs = [args.prompt] * 4  # generate 4 samples once
#     # uncondition_text_inputs = ['<unconditional><boi>'] * len(text_inputs)
#     # for i in range(len(text_inputs)):
#     #     text_inputs[i] = text_inputs[i] + ' Generate an image based on this description.<boi>'
#     #
#     # if guidance_scale > 1:
#     #     model_inputs = tokenizer(text_inputs + uncondition_text_inputs, return_tensors="pt", padding=True).to("cuda:0")
#     # else:
#     #     model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to("cuda:0")
#     with torch.no_grad():
#         sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
#         # input_ids = model_inputs['input_ids']
#         cur_len = input_ids.shape[1]
#         model_kwargs = {'attention_mask': attention_mask, 'use_cache': True}
#         model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

#         pred_tokens = []
#         for i in tqdm(range(1626)):
#             model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)

#             if i > 0 and guidance_scale > 1:
#                 outputs = vqllm(
#                     **model_inputs,
#                     return_dict=True,
#                     output_attentions=False,
#                     output_hidden_states=False,
#                 )
#             else:
#                 outputs = vqllm(
#                     **model_inputs,
#                     return_dict=True,
#                     output_attentions=False,
#                     output_hidden_states=False,
#                 )

#             next_token_logits = outputs.logits[:, -1:, :]

#             # if guidance_scale > 1:
#             #     cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
#             #     cfg_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
#             #     half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
#             #     pred_tokens.append(half_next_token)
#             #     next_token = torch.cat([half_next_token, half_next_token])


#             # else:
#             next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
#             pred_tokens.append(next_token)

#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_token], dim=-1)
#             model_kwargs = vqllm._update_model_kwargs_for_generation(
#                 outputs,
#                 model_kwargs,
#                 is_encoder_decoder=vqllm.config.is_encoder_decoder,
#             )

#         del sampling_kwargs
#         del model_inputs
#         del outputs
#         # 1. 拼接最终生成的序列
#         generated_ids = torch.cat(pred_tokens, dim=1)[0]  # shape: [1, T] → [T]

#         # 2. 获取特殊token id
#         boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
#         eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")

#         # 3. 找到 <boi> 和 <eoi> 的索引位置
#         # boi_pos = (generated_ids == boi_token_id).nonzero(as_tuple=False)
#         # eoi_pos = (generated_ids == eoi_token_id).nonzero(as_tuple=False)

#         # assert len(boi_pos) > 0 and len(eoi_pos) > 0, "未找到 <boi> 或 <eoi>"
#         # assert len(boi_pos) == len(eoi_pos), "boi 和 eoi 数量不一致"
#         boi_pos = np.arange(6)*271
#         # 4. 提取 boi-eoi 之间的 token（多个块）
#         vq_token_lists = []
#         for i in range(len(boi_pos)):
#             start = boi_pos[i] + 1
#             end = start + 256
#             vq_token = generated_ids[start:end]
#             vq_token_lists.append(vq_token)

#         pic_ori = os.path.basename(pic_path)
#         pic_num = pic_ori.split(".")[0]
#         pic_num = int(pic_num)
#         pic_dir = os.path.dirname(pic_path)
#         pred_vqcodes = torch.stack(vq_token_lists, dim=0).to("cuda")  # shape: [6, 256]
#         pred_vqcodes = pred_vqcodes - len(tokenizer)
#         pred_vqcodes = torch.clamp(pred_vqcodes, 0, 264191)
#         criterion = CrossEntropyLoss()
#         vocab_size = 264192
#         # 将预测 token 视作 logits 的 argmax，这里你需要 logits 才能真正计算 loss
#         # 假设你已经有生成时保存下来的 logits 列表 pred_logits，每个 logits 是 [1, vocab_size]
#         # 你应在生成时加个 pred_logits.append(next_token_logits.squeeze(1)) 来收集预测 logits

#         # 拼接成完整 token logits 序列（假设你生成了6×256个 token）
#         full_logits = torch.cat(pred_logits, dim=1)  # shape: [1, 6*256, vocab_size]
#         full_logits = full_logits.view(6, 256, vocab_size)  # shape: [6, 256, vocab_size]

#         # 计算 loss
#         loss = criterion(full_logits.view(-1, vocab_size), gt_img_tokens.view(-1))

#         # 5. 解码每个 VQ token 块为图像
#         for i, vq_token in enumerate(vq_token_lists):
#             # Step 1: 解码图像 token
#             vq_token = vq_token - len(tokenizer)
#             # vq_token = torch.clamp(vq_token, 0, 264191)
#             rec_img = image_tokenizer.pil_from_img_toks(vq_token,height=16,width=16)

#             # Step 2: 构造原图路径（根据 pic_num + i 命名）
#             k = pic_num + i
#             ori_path = os.path.join(pic_dir, f"{k:05d}.jpg")

#             # Step 3: 读取原图
#             if not os.path.exists(ori_path):
#                 print(f"⚠️ 原图不存在: {ori_path}")
#                 continue
#             ori_img = Image.open(ori_path).convert("RGB")

#             # Step 4: 尺寸对齐（如果需要）

#             ori_img = center_crop_image(ori_img,tgt_width=256,tgt_height=256)

#             # Step 5: 拼接图像（横向）
#             w, h = ori_img.size
#             combined = Image.new("RGB", (w * 2, h))
#             combined.paste(ori_img, (0, 0))
#             combined.paste(rec_img, (w, 0))

#             # Step 6: 保存拼接图
#             combined.save(f"{image_save_pth}/compare_{i}.jpg")

def main(args):
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

    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
    )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')

    vqgan_cfg_path = "model/vqgan_imagenet_f16_1024/configs/model.yaml"
    vqgan_ckpt_path = "model/vqgan_imagenet_f16_1024/ckpts/last.ckpt"
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0")

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
    input_ids, attention_mask = build_vqa_inference_input(tokenizer, sources)
    # input_ids = input_ids[:,:-256]
    # attention_mask = attention_mask[:,:-256]
    with torch.no_grad():
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        cur_len = input_ids.shape[1]
        model_kwargs = {'attention_mask': attention_mask, 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        pred_tokens = []
        pred_logits = []
        # image_insert_pos = [269 * i for i in range(6)]
        image_insert_pos = []
        boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        num_img_tokens = 256
        generating_image_tokens = False
        image_tokens_remaining = 0
        next_token = torch.tensor([[0]]).to("cuda")
        for i in tqdm(range(1617)):
            model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = vqllm(**model_inputs, return_dict=True)

            next_token_logits = outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
            next_token_logits = next_token_logits[:, :, :264192]
            ###############################################################
            # in_image_range = any(p <= i < p + 256 for p in image_insert_pos)
            #
            # # 找出最大值和对应索引
            # # max_prob, max_idx = torch.max(probs, dim=-1)  # [1, 1]
            # # logits_flat = next_token_logits.view(-1)  # 展平成 [vocab_size]
            # # max_val, max_idx = torch.max(logits_flat, dim=0)
            # # print("logit:", logits_flat)
            # # print("最大值:", max_val.item())
            # # print("最大值索引（token id）:", max_idx.item())
            # if in_image_range:
            #     next_token, _ = sample(next_token_logits, **sampling_kwargs)
            # else:
            #     logits = next_token_logits[:, :, :256000]  # 只保留前256000个token的logits
            #     probs = F.softmax(logits, dim=-1)  # [1, 1, 256000]
            #
            #     # 找出最大值和对应索引
            #     max_prob, max_idx = torch.max(probs, dim=-1)  # [1, 1]
            #     next_token = max_idx
            ################################################################
            if generating_image_tokens:
                next_token, _ = sample(next_token_logits, **sampling_kwargs)
                image_tokens_remaining -= 1
                print("nextToken1:", next_token)
                if image_tokens_remaining == 0:
                    generating_image_tokens = False


            else:

                pre_token = next_token

                # 普通文本逻辑，只保留前256000个token
                logits = next_token_logits[:, :, :256000]
                probs = F.softmax(logits, dim=-1)
                max_prob, max_idx = torch.max(probs, dim=-1)
                next_token = max_idx
                # next_token, _ = sample_lw(logits, **sampling_kwargs)

                # 如果输出了 <boi>，进入图像生成状态
                if next_token.item() == boi_token_id:
                    generating_image_tokens = True
                    image_tokens_remaining = num_img_tokens
                    image_insert_pos.append(i)
                if pre_token[0]>256000:
                    next_token = torch.tensor([[8]]).to("cuda")
                    # continue
                print("nextToken2:", next_token)

                # if i in [x - 1 for x in image_insert_pos]:
                #     next_token = torch.tensor([[7]]).to("cuda")  # <boi>
                # elif i in [x + 256 for x in image_insert_pos]:
                #     next_token = torch.tensor([[8]]).to("cuda")  # <eoi>

            pred_tokens.append(next_token)
            pred_logits.append(next_token_logits)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            # print("input_ids:", input_ids)
            input_ids_ = input_ids.squeeze(0)  # 变成 1D tensor [N]
            with open("input_ids.txt", "w", encoding="utf-8") as f:
                for token_id in input_ids_:
                    f.write(f"{token_id.item()}\n")  # 每个元素一行
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
        generated_ids = torch.cat(pred_tokens, dim=1)[0]  # [T]
        full_logits = torch.cat(pred_logits, dim=1)  # [1, T, vocab_size]

        # ====== 提取图像 token 段 ======
        # boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        # # print("gen:", generated_ids)
        # print("boi:", boi_token_id)
        # boi_pos = (generated_ids == boi_token_id).nonzero(as_tuple=True)[0]
        count = (generated_ids < 256000).sum().item()
        print("小于 256000 的数量为：", count)
        # # 确保找到6个<boi>标记
        # assert len(boi_pos) == 6, f"Expected 6 <boi> tokens, found {len(boi_pos)}"
        # boi_pos = np.arange(6) * 269+3
        img_logits = []
        for pos in image_insert_pos:
            start = pos
            end = start + 256
            img_logits.append(full_logits[0, start:end])  # 每张256个 token

        img_logits = torch.stack(img_logits, dim=0)  # [6, 256, vocab]
        criterion = CrossEntropyLoss()

        # 确保维度匹配
        assert img_logits.shape[0] * img_logits.shape[1] == gt_img_tokens.numel(), \
            f"Shape mismatch: img_logits {img_logits.shape} vs gt_img_tokens {gt_img_tokens.shape}"
        print("img shape: ", img_logits.shape)
        # image_ce_loss = criterion(img_logits.reshape(-1, 264192), gt_img_tokens.view(-1))
        # image_ce_loss = criterion(img_logits.reshape(-1, 264192)[:, -8192:], gt_img_tokens.view(-1))
        # print("📉 Average Image CrossEntropy Loss:", image_ce_loss.item())

        image_losses = []
        img_loss_l = img_logits.reshape(-1, 264192)[:, -8192:-7168]

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
            start = image_insert_pos[i]
            end = start + 256
            vq_token = generated_ids[start:end]
            vq_token_lists.append(vq_token)

        # pic_ori = os.path.basename(pic_path)
        # pic_num = int(pic_ori.split(".")[0])
        # pic_dir = os.path.dirname(pic_path)
        pred_vqcodes = torch.stack(vq_token_lists, dim=0).to("cuda")  # [6, 256]
        pred_vqcodes = pred_vqcodes - len(tokenizer)
        pred_vqcodes = torch.clamp(pred_vqcodes, 0, 1023)
        future_vqcodes = torch.stack(future_vqcodes, dim=0).to("cuda")
        for i, vq_token in enumerate(pred_vqcodes):
            # vq_token = vq_token - len(tokenizer)
            print("vq_token:", vq_token.shape)
            with open("vq.txt", "w", encoding="utf-8") as f:
                for token_id in vq_token:
                    f.write(f"{token_id.item()}\n")  # 每个元素一行
            rec_img = image_tokenizer.pil_from_img_toks(vq_token, height=16, width=16)
            ori_img = image_tokenizer.pil_from_img_toks(future_vqcodes[i], height=16, width=16)
            # k = pic_num + i
            # ori_path = os.path.join(pic_dir, f"{k:05d}.jpg")
            # if not os.path.exists(ori_path):
            #     print(f"⚠️ 原图不存在: {ori_path}")
            #     continue

            # ori_img = Image.open(ori_path).convert("RGB")
            # ori_img = center_crop_image(ori_img, tgt_width=256, tgt_height=256)

            w, h = ori_img.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(ori_img, (0, 0))
            combined.paste(rec_img, (w, 0))
            combined.save(f"{image_save_pth}/compare_{i}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='Junfeng5/Liquid_V1_7B',
                        help='model path, default to huggingface repo id')
    parser.add_argument('--save_path', type=str, default='samples/t2i', help='save path')
    parser.add_argument('--prompt', type=str, help='input text prompt')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit to save memory')
    parser.add_argument('--cfg', type=float, default=7.0, help='Classifier-Free Guidance scale')
    parser.add_argument('--TopP', type=float, default=0.96, help='Top P, max=1.0')
    parser.add_argument('--TopK', type=int, default=512, help='Top K, max=264192')
    parser.add_argument('--temperature', type=float, default=0.2, help='sampling temperature, max=1.0')

    args = parser.parse_args()
    main(args)
