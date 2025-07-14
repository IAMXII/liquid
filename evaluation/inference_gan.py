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
import json

def format_wp(wp):  # 格式化为 "[x,y]"
    return f"[{wp[0]:.2f},{wp[1]:.2f}]"

def build_vqa_inference_input(tokenizer, sources):
    tokenizer_len = len(tokenizer)

    def load_vqcode(s):
        return torch.tensor(json.loads(s)) + tokenizer_len

    known_vqcodes = [load_vqcode(s) for s in sources["known_vqcodes"]]
    future_vqcodes = [load_vqcode(s) for s in sources["future_vqcodes"]]
    known_wps = sources["known_waypoints"]

    def make_vqtext(wp):
        return "<boi><eoi>" + format_wp(wp)

    human_text = "We already know three frames and their waypoints: " + ", ".join(
        [make_vqtext(wp) for wp in known_wps]
    ) + ", now predict the next frame, their waypoints are "

    def insert_vqcodes(input_ids, vqcode_list):
        output = []
        i = 0
        vq_iter = iter(vqcode_list)
        while i < len(input_ids):
            token = input_ids[i].item()
            token_str = tokenizer.convert_ids_to_tokens(token)
            if token_str == "<boi>":
                output.append(token)
                i += 1
                vq = next(vq_iter)  # vq: [num_codebooks, sequence_length]，例如 [8, 1024]

                # ✅ 平铺展开 VQ token：先按 codebook，再按位置（或按位置再按 codebook）
                # 通常按顺序：先 codebook0 的 0..1023，codebook1 的 0..1023 ...
                # vq_flat = vq.contiguous().view(-1).tolist()  # [8 * 1024] → list[int]

                output.extend(vq)

                # 跳到 <eoi>
                while i < len(input_ids):
                    if tokenizer.convert_ids_to_tokens(input_ids[i].item()) == "<eoi>":
                        output.append(input_ids[i].item())
                        i += 1
                        break
                    i += 1
            else:
                output.append(token)
                i += 1
        return torch.tensor(output, dtype=torch.long)

    from liquid import conversation as conversation_lib
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], human_text)
    conv.append_message(conv.roles[1], "")  # 模型生成
    prompt = conv.get_prompt()

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,padding=True).input_ids[0]
    attention_mask = tokenizer(prompt, return_tensors="pt", truncation=True,padding=True)["attention_mask"]

    # 插入 VQ token（历史3帧 + 待预测图像）
    all_vqcodes = known_vqcodes + future_vqcodes
    input_ids = insert_vqcodes(input_ids, all_vqcodes, tokenizer)

    return input_ids.unsqueeze(0), attention_mask # [1, T]

def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    Width,Height = ori_image.size
    factor = min(Width,Height)/min(tgt_width,tgt_height)
    input_image = ori_image.resize((int(Width/factor),int(Height/factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size   # Get dimensions

    left = (resize_width - tgt_width)//2
    top = (resize_height - tgt_height)//2
    right = (resize_width + tgt_width)//2
    bottom = (resize_height + tgt_height)//2
    # Crop the center of the image
    input_image = input_image.crop((left, top, right, bottom))
    return input_image


def main(args):
    temperature = args.temperature
    guidance_scale = args.cfg
    top_K = args.TopK
    top_P = args.TopP
    image_save_pth = args.save_path
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth)

    assert temperature <= 1.0
    assert top_K <= 8192
    assert top_P <= 1.0

    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    ori_vocabe_size = len(tokenizer)

    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
    )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')
    vqgan_cfg_path = "chameleon/vqgan.yaml"
    vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0", )
    with open("000000.jsonl", "r", encoding="utf-8") as f:
        first_line = f.readline()
        sources = json.loads(first_line)
    pic_path = sources["pic_path"]
    input_ids,attention_mask = build_vqa_inference_input(tokenizer, sources).to("cuda")

    # text_inputs = [args.prompt] * 4  # generate 4 samples once
    # uncondition_text_inputs = ['<unconditional><boi>'] * len(text_inputs)
    # for i in range(len(text_inputs)):
    #     text_inputs[i] = text_inputs[i] + ' Generate an image based on this description.<boi>'
    #
    # if guidance_scale > 1:
    #     model_inputs = tokenizer(text_inputs + uncondition_text_inputs, return_tensors="pt", padding=True).to("cuda:0")
    # else:
    #     model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        # input_ids = model_inputs['input_ids']
        cur_len = input_ids.shape[1]
        model_kwargs = {'attention_mask': attention_mask, 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        pred_tokens = []
        for i in tqdm(range(1626)):
            model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if i > 0 and guidance_scale > 1:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            else:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            next_token_logits = outputs.logits[:, -1:, :]

            if guidance_scale > 1:
                cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                cfg_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
                half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                pred_tokens.append(half_next_token)
                next_token = torch.cat([half_next_token, half_next_token])


            else:
                next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                pred_tokens.append(next_token)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

        del sampling_kwargs
        del model_inputs
        del outputs
        # 1. 拼接最终生成的序列
        generated_ids = torch.cat(pred_tokens, dim=1)[0]  # shape: [1, T] → [T]

        # 2. 获取特殊token id
        boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")

        # 3. 找到 <boi> 和 <eoi> 的索引位置
        boi_pos = (generated_ids == boi_token_id).nonzero(as_tuple=False)
        eoi_pos = (generated_ids == eoi_token_id).nonzero(as_tuple=False)

        assert len(boi_pos) > 0 and len(eoi_pos) > 0, "未找到 <boi> 或 <eoi>"
        assert len(boi_pos) == len(eoi_pos), "boi 和 eoi 数量不一致"

        # 4. 提取 boi-eoi 之间的 token（多个块）
        vq_token_lists = []
        for i in range(len(boi_pos)):
            start = boi_pos[i].item() + 1
            end = eoi_pos[i].item()
            vq_token = generated_ids[start:end]
            vq_token_lists.append(vq_token)

        pic_ori = os.path.basename(pic_path)
        pic_num = pic_ori.split(".")[0]
        pic_num = int(pic_num)
        pic_dir = os.path.dirname(pic_path)
        # 5. 解码每个 VQ token 块为图像
        for i, vq_token in enumerate(vq_token_lists):
            # Step 1: 解码图像 token
            vq_token = vq_token - len(tokenizer)
            vq_token = torch.clamp(vq_token, 0, 8191)
            rec_img = image_tokenizer.pil_from_img_toks(vq_token)

            # Step 2: 构造原图路径（根据 pic_num + i 命名）
            k = pic_num + i
            ori_path = os.path.join(pic_dir, f"{k:05d}.jpg")

            # Step 3: 读取原图
            if not os.path.exists(ori_path):
                print(f"⚠️ 原图不存在: {ori_path}")
                continue
            ori_img = Image.open(ori_path).convert("RGB")

            # Step 4: 尺寸对齐（如果需要）

            ori_img = center_crop_image(ori_img,tgt_width=256,tgt_height=256)

            # Step 5: 拼接图像（横向）
            w, h = ori_img.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(ori_img, (0, 0))
            combined.paste(rec_img, (w, 0))

            # Step 6: 保存拼接图
            combined.save(f"{image_save_pth}/compare_{i}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='Junfeng5/Liquid_V1_7B',
                        help='model path, default to huggingface repo id')
    parser.add_argument('--save_path', type=str, default='samples/t2i', help='save path')
    parser.add_argument('--prompt', type=str, required=True, help='input text prompt')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit to save memory')
    parser.add_argument('--cfg', type=float, default=7.0, help='Classifier-Free Guidance scale')
    parser.add_argument('--TopP', type=float, default=0.96, help='Top P, max=1.0')
    parser.add_argument('--TopK', type=int, default=4096, help='Top K, max=8192')
    parser.add_argument('--temperature', type=float, default=0.99, help='sampling temperature, max=1.0')

    args = parser.parse_args()
    main(args)


