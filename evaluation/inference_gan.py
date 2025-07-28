# -*- coding: utf-8 -*-
"""
基于 VQGAN 编码的图像生成与 VQA 推理脚本：
- 输入包含图像 token（VQCode）与 Waypoint，生成图像 token。
- 支持采样策略（TopK/TopP/Temperature），并评估生成图像 token 的 CrossEntropy Loss。
"""

import os
import json
import time
import argparse
import torch
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
from threading import Thread
from torch.nn import functional as F, CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms

from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import conv_templates
from T2I_Eval.genaibench_generation import sample

# ====================== Token采样函数 ======================
def sample_lw(logits, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
    """
    从 logits 中采样 token（支持 TopK / TopP / 温度 / 贪婪解码）
    """
    logits = logits[:, -1, :]
    if temperature != 1.0:
        logits = logits / temperature

    # Top-K 筛选
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

    # Top-P 筛选（Nucleus Sampling）
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return (torch.multinomial(probs, 1) if sample_logits else torch.argmax(probs, dim=-1, keepdim=True)), probs


def format_wp(wp):
    return f"[{wp[0]:.2f},{wp[1]:.2f}]"


# ====================== 构建输入函数 ======================
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

    human_text = "We already know three frames and their waypoints: " + \
        ", ".join([make_vqtext(wp) for wp in known_wps]) + \
        ", now predict the next frames, their waypoints are " + \
        ", ".join([format_wp(wp) for wp in future_wps])

    from liquid import conversation as conversation_lib
    conv = conversation_lib.conv_templates["gemma"].copy()
    conv.append_message(conv.roles[0], human_text)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    def insert_vqcodes_with_mask(input_ids, attention_mask, vqcode_list):
        output_ids, output_mask = [], []
        i, vq_iter = 0, iter(vqcode_list)
        while i < len(input_ids):
            token = input_ids[i]
            token_str = tokenizer.convert_ids_to_tokens(token)
            if token_str == "<boi>":
                output_ids.append(token)
                output_mask.append(0)
                i += 1
                vq = next(vq_iter).tolist()
                output_ids.extend(vq)
                output_mask.extend([1] * len(vq))
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
        return torch.tensor(output_ids), torch.tensor(output_mask)

    input_ids, attention_mask = insert_vqcodes_with_mask(input_ids, attention_mask, known_vqcodes + future_vqcodes)
    return input_ids.unsqueeze(0).to("cuda"), attention_mask.unsqueeze(0).to("cuda")


# ====================== 图像中心裁剪函数 ======================
def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    Width, Height = ori_image.size
    factor = min(Width, Height) / min(tgt_width, tgt_height)
    input_image = ori_image.resize((int(Width / factor), int(Height / factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size
    left = (resize_width - tgt_width) // 2
    top = (resize_height - tgt_height) // 2
    right = (resize_width + tgt_width) // 2
    bottom = (resize_height + tgt_height) // 2
    return input_image.crop((left, top, right, bottom))


# ====================== 模型加载辅助函数 ======================
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
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# ====================== 主流程入口 ======================
def main(args):
    # 参数设置
    temperature = args.temperature
    top_K = args.TopK
    top_P = args.TopP
    image_save_pth = args.save_path
    os.makedirs(image_save_pth, exist_ok=True)

    # 加载模型与 tokenizer
    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
    )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')

    # 初始化图像tokenizer（VQGAN）
    image_tokenizer = ImageTokenizer(
        cfg_path="model/vqgan_imagenet_f16_1024/configs/model.yaml",
        ckpt_path="model/vqgan_imagenet_f16_1024/ckpts/last.ckpt",
        device="cuda:0")

    # 加载一条测试数据（第111行）
    with open("/data/tempdata_val/000000.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 110:
                sources = json.loads(line)
                break

    future_vqcodes = [torch.tensor(json.loads(s)) for s in sources["future_vqcodes"]]
    gt_img_tokens = torch.cat(future_vqcodes, dim=0).to("cuda")  # [6*256]

    # 构造输入
    input_ids, attention_mask = build_vqa_inference_input(tokenizer, sources)

    # 推理阶段：逐 token 自回归采样
    with torch.no_grad():
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        cur_len = input_ids.shape[1]
        model_kwargs = {'attention_mask': attention_mask, 'use_cache': True, 'cache_position': torch.arange(cur_len).to("cuda")}

        pred_tokens = []
        pred_logits = []
        image_insert_pos = []
        boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        num_img_tokens = 256
        generating_image_tokens = False
        image_tokens_remaining = 0
        next_token = torch.tensor([[0]]).to("cuda")

        for i in tqdm(range(1617)):
            model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = vqllm(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1:, :][:, :, :264192]  # 截断 vocab 范围

            if generating_image_tokens:
                next_token, _ = sample(next_token_logits, **sampling_kwargs)
                image_tokens_remaining -= 1
                if image_tokens_remaining == 0:
                    generating_image_tokens = False
            else:
                pre_token = next_token
                logits = next_token_logits[:, :, :256000]
                next_token, _ = sample_lw(logits, **sampling_kwargs)
                if next_token.item() == boi_token_id:
                    generating_image_tokens = True
                    image_tokens_remaining = num_img_tokens
                    image_insert_pos.append(i)
                if pre_token[0] > 256000:
                    next_token = torch.tensor([[8]]).to("cuda")  # fallback to <eoi>

            pred_tokens.append(next_token)
            pred_logits.append(next_token_logits)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            model_kwargs = vqllm._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

        # 拼接所有生成 token
        generated_ids = torch.cat(pred_tokens, dim=1)[0]
        full_logits = torch.cat(pred_logits, dim=1)

        # 提取 image logits 段落，计算 loss
        img_logits = torch.stack([full_logits[0, p:p+256] for p in image_insert_pos], dim=0)  # [6, 256, vocab]
        criterion = CrossEntropyLoss()
        img_loss_l = img_logits.reshape(-1, 264192)[:, -8192:-7168]  # 可调区间

        image_losses = []
        for i in range(6):
            start, end = i * 256, (i + 1) * 256
            logits_i = img_loss_l[start:end, :]
            targets_i = gt_img_tokens.view(-1)[start:end]
            loss_i = criterion(logits_i, targets_i)
            image_losses.append(loss_i.item())
            print(f"📉 Image {i} CrossEntropy Loss:", loss_i.item())

        avg_loss = sum(image_losses) / len(image_losses)
        print("📉 Average Image CrossEntropy Loss:", avg_loss)

        # 解码生成图像并保存对比图
        pred_vqcodes = torch.stack([generated_ids[p:p+256] for p in image_insert_pos], dim=0).to("cuda")
        pred_vqcodes = torch.clamp(pred_vqcodes - len(tokenizer), 0, 1023)
        future_vqcodes = torch.stack(future_vqcodes, dim=0).to("cuda")

        for i, vq_token in enumerate(pred_vqcodes):
            rec_img = image_tokenizer.pil_from_img_toks(vq_token, height=16, width=16)
            ori_img = image_tokenizer.pil_from_img_toks(future_vqcodes[i], height=16, width=16)
            w, h = ori_img.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(ori_img, (0, 0))
            combined.paste(rec_img, (w, 0))
            combined.save(f"{image_save_pth}/compare_{i}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Junfeng5/Liquid_V1_7B')
    parser.add_argument('--save_path', type=str, default='samples/t2i')
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--cfg', type=float, default=7.0)
    parser.add_argument('--TopP', type=float, default=0.96)
    parser.add_argument('--TopK', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.99)
    args = parser.parse_args()
    main(args)
