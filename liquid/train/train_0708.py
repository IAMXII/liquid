# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import numpy as np

import transformers
import tokenizers

from liquid.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                              DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from torch.utils.data import Dataset
from liquid.train.llava_trainer import LLaVATrainer

from liquid import conversation as conversation_lib
from liquid.model import *
from liquid.mm_utils import tokenizer_image_token

from PIL import Image
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from datasets import load_from_disk, concatenate_datasets
from liquid.model.language_model.mini_gemini_gemma import MiniGeminiGemmaForCausalLM
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def format_wp(wp):
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

    gpt_text = ", ".join([make_vqtext(wp) for wp in future_wps])

    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], human_text)
    conv.append_message(conv.roles[1], gpt_text)
    prompt = conv.get_prompt()

    input_ids = \
    tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).input_ids[0]
    print("input_ids: \n",input_ids)
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

    return input_ids, labels


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mmprojector: bool = field(default=False)
    reload_embedder: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_aux: Optional[str] = field(default=None)  # auxiliary vision tower
    optimize_vision_tower: bool = field(default=False)  # whether to optimize vision tower
    optimize_vision_tower_aux: bool = field(default=False)  # whether to optimize auxiliary vision tower
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrained_mmprojector: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    vq_resolution: int = 256,
    vqconversion: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_size_aux: Optional[int] = field(default=320)
    image_grid: Optional[int] = field(default=1)
    image_global: Optional[bool] = field(default=False)
    t2i_prompt_type: Optional[str] = None,
    cfg_ratio: Optional[float] = 0.85,
    percentage: Optional[str] = field(default='1.0')
    T2I_ratio: Optional[float] = field(default=0.5)
    shuffleseed: Optional[int] = field(default=42)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    label_smoothing_factor: float = 0.0


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # import pdb;pdb.set_trace()
    keys_to_match = ['mm_projector', 'attn_projection']
    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        print('Only save Adapter')
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_uni']
        # add vision tower
        keys_to_match.extend(['vision_tower'])
        # add vision tower aux
        keys_to_match.extend(['vision_fpn', 'vision_stages'])
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # not included <|im_end|>
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # include <|im_end|> for all rounds
            # if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
            if getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain_guided(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        prompt: str = None,
) -> Dict:
    # add end signal and concatenate together
    guided_prompt = []
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        guided_prompt.append(source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', ''))
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets, prompt=guided_prompt)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        prompt: str = None,
        refine_prompt: bool = False,
        vq_resolution: int = 256,
        t2i_prompt_type: str = None,
        cfg_ratio: float = 0.85,
        vqconversion: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("plain_guided"):
        return preprocess_plain_guided(sources, tokenizer, prompt=prompt)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("gemma"):
        return preprocess_gemma(sources, tokenizer, has_image=has_image, vq_resolution=vq_resolution,
                                t2i_prompt_type=t2i_prompt_type, cfg_ratio=cfg_ratio, vqconversion=vqconversion)

    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        # list_data_dict = json.load(open(data_path, "r"))
        if data_path[-5:] == 'jsonl':
            list_data_dict = []
            with open(os.path.join(data_path), 'r', encoding="utf-8") as f:
                for line in f:
                    list_data_dict.append(json.loads(line))
        elif '^^' in data_path:
            data_path_list = data_path.split('^^')
            list_data_dict = []
            for data_path_i in data_path_list:
                all_files = os.listdir(data_path_i)
                all_files.sort()
                # import time
                # startTime =  time.time()
                for file_name in all_files:
                    with open(os.path.join(data_path_i, file_name), 'r', encoding="utf-8") as f:
                        for line in f:
                            list_data_dict.append(json.loads(line))
        else:
            list_data_dict = []
            all_files = os.listdir(data_path)
            all_files.sort()
            # import time
            # startTime =  time.time()
            for file_name in all_files:
                with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as f:
                    for line in f:
                        # data = json.loads(line)
                        # vqres = "vqcode_{}".format(str(data_args.vq_resolution))
                        # used_data = {vqres:data[vqres],  "caption-InternVL1.5":data[ "caption-InternVL1.5"]}
                        # if "text" in data:
                        #     used_data["text"] = data["text"]
                        list_data_dict.append(json.loads(line))
                        # list_data_dict.append(used_data)

        rank0_print("Formatting inputs...Skip in lazy mode")
        print('all data: ', len(list_data_dict))
        print('use t2i prompt type:', data_args.t2i_prompt_type)
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.vqconversion = data_args.vqconversion
        self.cfg_ratio = data_args.cfg_ratio
        self.vq_resolution = data_args.vq_resolution
        self.t2i_prompt_type = data_args.t2i_prompt_type

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempt, max_attempt = 0, 10
        # data_dict = self._sample_item(i)
        while attempt < max_attempt:
            try:
                # sample an item
                data_dict = self._sample_item(i)
                break
            except:
                attempt += 1
                print(f"Error in loading {i}, retrying...")
                print(self.list_data_dict[i])
                i = random.randint(0, len(self.list_data_dict) - 1)

        return data_dict

    def _sample_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            vq_resolution=self.vq_resolution,
            t2i_prompt_type=self.t2i_prompt_type,
            cfg_ratio=self.cfg_ratio,
            vqconversion=self.vqconversion)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # import pdb;pdb.set_trace()

        t2i_prompt_type = self.data_args.t2i_prompt_type
        vqconversion = self.data_args.vqconversion
        vq_resolution = self.data_args.vq_resolution
        T2I_ratio = self.data_args.T2I_ratio

        processed_instances = []
        for sources in instances:
            # print(sources['data_type'])
            if sources['data_type'] in ['T2I']:  # :  # image text pair
                eoi = torch.tensor([4])
                boi = torch.tensor([3])
                eos = torch.tensor([2])
                bos = torch.tensor([1])
                image_token_index = torch.tensor([IMAGE_TOKEN_INDEX])
                if np.random.rand() < T2I_ratio:  # T2I mode
                    prompt = ' Generate an image based on this description.'

                    if sources['text'] != 'no' and sources['InVL_caption'] != 'no':
                        if np.random.rand() > 0.8:
                            text = sources['text'] + prompt
                        else:  # have more samples in InternVL long caption
                            text = sources['InVL_caption'] + prompt
                    elif sources['text'] != 'no':
                        text = sources['text'] + prompt
                    elif sources['InVL_caption'] != 'no':
                        text = sources['InVL_caption'] + prompt
                    else:
                        text = '<unconditional>'  # <0x02>   # <reserved11111> as unconditional token for Chameleon

                    if np.random.rand() > 0.9:
                        text = "<unconditional>"

                    conversations = [text]

                    input_ids = self.tokenizer(
                        conversations,
                        return_tensors="pt",
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                    ).input_ids[0]
                    # import pdb;pdb.set_trace()
                    if len(input_ids) > 1750:
                        print('input truncated, original len: ', len(input_ids))
                        input_ids = input_ids[:1750]
                    instruction_len = len(input_ids)
                    # import pdb;pdb.set_trace()
                    vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
                    vqcode = torch.tensor([vqcode])  # + len(self.tokenizer)
                    # vqcode = vqcode.to(input_ids) + len(tokenizer)
                    input_ids = torch.cat([input_ids, boi, image_token_index, eoi, eos])
                    cur_data_type = 1  # T2I

                # import pdb;pdb.set_trace()
                targets = input_ids.clone()
                targets[: instruction_len] = IGNORE_INDEX
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=targets,
                    image=vqcode,
                    data_type=cur_data_type
                ))
            elif sources['data_type'] == 'waypoint_vqa':
                input_ids, labels = build_vqa_pair_with_vqcode(self.tokenizer, sources)
                known_vqcodes = [torch.tensor(json.loads(s)) for s in sources["known_vqcodes"]]
                future_vqcodes = [torch.tensor(json.loads(s)) for s in sources["future_vqcodes"]]
                # print(known_vqcodes[0])
                vqcode = known_vqcodes + future_vqcodes
                # vqcode = torch.tensor(vqcode)
                vqcode = torch.stack(vqcode, dim=0)
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=labels,
                    image=vqcode,
                    data_type=5
                ))
            elif sources['data_type'] in ['I2T']:
                eoi = torch.tensor([4])
                boi = torch.tensor([3])
                eos = torch.tensor([2])
                bos = torch.tensor([1])
                image_token_index = torch.tensor([IMAGE_TOKEN_INDEX])

                # instruct = 'Detailed long caption of this image is:'
                caption = sources['InVL_caption']

                caption_ids = self.tokenizer(caption, return_tensors="pt", padding="longest",
                                             max_length=self.tokenizer.model_max_length, truncation=True, ).input_ids[0]
                vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
                vqcode = torch.tensor([vqcode])  # + len(self.tokenizer)
                input_ids = torch.cat([bos, boi, image_token_index, eoi, caption_ids[1:], eos])
                instruction_len = len(input_ids) - len(caption_ids)
                cur_data_type = 2  # I2T

                # import pdb;pdb.set_trace()
                targets = input_ids.clone()
                targets[: instruction_len] = IGNORE_INDEX
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=targets,
                    image=vqcode,
                    data_type=cur_data_type
                ))
            elif sources['data_type'] in ['mllm_pretrain']:  # mllm pretrain dataset use plain conversation
                conversations = []
                source = json.loads(sources['text'])
                vqcode = sources['vqcode_256']
                assert len(source) == 2
                assert DEFAULT_IMAGE_TOKEN in source[0]['value']
                source[0]['value'] = DEFAULT_IMAGE_TOKEN
                conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
                conversations.append(conversation)
                # tokenize conversations
                has_image = vqcode != 'no'
                if has_image:
                    input_ids = torch.stack(
                        [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in
                         conversations], dim=0)
                    vqcode = json.loads(vqcode)
                    vqcode = torch.tensor([vqcode])  # [bz,4,256] # + len(self.tokenizer)
                else:
                    input_ids = self.tokenizer(conversations, return_tensors="pt", padding="longest",
                                               max_length=self.tokenizer.model_max_length, truncation=True, ).input_ids
                    vqcode = []

                targets = input_ids.clone()
                # import pdb;pdb.set_trace()
                for target, source in zip(targets, [source]):
                    tokenized_len = len(tokenizer_image_token(source[0]['value'], self.tokenizer))
                    target[:tokenized_len] = IGNORE_INDEX
                # import pdb;pdb.set_trace()
                input_ids = input_ids[0]
                targets = targets[0]
                if has_image:
                    cur_data_type = 2  # I2T
                else:
                    cur_data_type = 0  # text only
                # import pdb;pdb.set_trace()
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=targets,
                    image=vqcode,
                    data_type=cur_data_type
                ))
            elif sources['data_type'] in ['mllm_finetune']:  # mllm dataset
                conv = conversation_lib.default_conversation.copy()
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

                vqcode = sources['vqcode_256']
                has_image = vqcode != 'no'

                # Apply prompt templates
                conversations = []
                for i, source in enumerate([json.loads(sources['text'])]):
                    if roles[source[0]["from"]] != conv.roles[0]:
                        # Skip the first one if it is not from human
                        source = source[1:]

                    conv.messages = []
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == conv.roles[j % 2], f"{i}"
                        conv.append_message(role, sentence["value"])
                    conversations.append(conv.get_prompt())

                # Tokenize conversations
                if has_image:
                    input_ids = torch.stack(
                        [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in
                         conversations], dim=0)
                    vqcode = json.loads(vqcode)
                    vqcode = torch.tensor([vqcode])  # [bz,4,256] # + len(self.tokenizer)
                else:
                    input_ids = self.tokenizer(
                        conversations,
                        return_tensors="pt",
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                    ).input_ids
                    vqcode = []
                targets = input_ids.clone()

                assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
                # Mask targets
                sep = conv.sep + conv.roles[1] + ": "
                for conversation, target in zip(conversations, targets):
                    total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

                    rounds = conversation.split(conv.sep2)
                    cur_len = 1
                    target[:cur_len] = IGNORE_INDEX
                    for i, rou in enumerate(rounds):
                        if rou == "":
                            break

                        parts = rou.split(sep)
                        if len(parts) != 2:
                            break
                        parts[0] += sep

                        if has_image:
                            round_len = len(tokenizer_image_token(rou, self.tokenizer))
                            instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2
                        else:
                            round_len = len(self.tokenizer(rou).input_ids)
                            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                        if i != 0 and not getattr(self.tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                            round_len -= 1
                            instruction_len -= 1

                        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                        cur_len += round_len
                    target[cur_len:] = IGNORE_INDEX
                    if cur_len < self.tokenizer.model_max_length:
                        if cur_len != total_len:
                            target[:] = IGNORE_INDEX
                            print(
                                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                                f" (ignored)"
                            )
                input_ids = input_ids[0]
                targets = targets[0]
                if has_image:
                    cur_data_type = 2  # I2T
                else:
                    cur_data_type = 0  # text only
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=targets,
                    image=vqcode,
                    data_type=cur_data_type
                ))

            else:  # text pretrain mode
                text = sources['text']
                assert text != 'no'
                input_ids = \
                    self.tokenizer(text, return_tensors="pt", padding="longest",
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=True, ).input_ids[0]
                targets = input_ids.clone()
                processed_instances.append(dict(
                    input_ids=input_ids,
                    labels=targets,
                    image=[],
                    data_type=0,  # text only
                ))

        ### batching ...
        # import pdb;pdb.set_trace()
        input_ids, labels = tuple([instance[key] for instance in processed_instances]
                                  for key in ("input_ids", "labels"))
        # import pdb;pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        data_types = torch.tensor([instance['data_type'] for instance in processed_instances])

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            data_types=data_types,
        )
        # import pdb;pdb.set_trace()
        if 'image' in processed_instances[0]:
            images = [instance['image'] for instance in processed_instances]
            batch['images'] = images
        if 'image_aux' in instances[0]:
            images = [instance['image_aux'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images) and len(images) > 1:
                batch['images_aux'] = torch.stack(images)
            else:
                batch['images_aux'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_path = data_args.data_path
    percentage = data_args.percentage
    shuffleseed = data_args.shuffleseed

    if '^^' in data_path:
        data_paths = data_path.split('^^')
        if '^^' in percentage:
            percentages = [float(p) for p in percentage.split('^^')]
        else:
            percentages = [float(percentage)] * len(data_paths)
        assert len(percentages) == len(data_paths)

        hgdata_list = []
        print('loading subsets...')
        for percent, hgdata_path in zip(percentages, data_paths):
            # import pdb;pdb.set_trace()
            subset = load_from_disk(hgdata_path)
            sub_len = subset.num_rows
            subset = subset.select(range(int(sub_len * percent)))

            hgdata_list.append(subset)
        train_dataset = concatenate_datasets(hgdata_list)
        if shuffleseed != 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)

        # print('to iterable...')
        # iterable_dataset = concat_dataset.to_iterable_dataset(num_shards=4096)
        # print('shuffling...')
        # train_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)
        print(hgdata_list)
    else:
        print('loading subsets...')
        train_dataset = load_from_disk(data_path)
        sub_len = train_dataset.num_rows
        percentage = float(percentage)
        train_dataset = train_dataset.select(range(int(sub_len * percentage)))
        if shuffleseed != 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)

    # import pdb;pdb.set_trace()
    print('training samples: ', train_dataset.num_rows)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    print("local_rank", local_rank)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    if model_args.vision_tower is not None:
        if "mistral" in model_args.model_name_or_path:
            model = MiniGeminiMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        elif "mixtral" in model_args.model_name_or_path:
            model = MiniGeminiMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            from deepspeed.utils import set_z3_leaf_modules
            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "gemma" in model_args.model_name_or_path:
            # import pdb;pdb.set_trace()
            model = MiniGeminiGemmaForCausalLM.from_pretrained( ### liuwei
            # model = MiniGeminiLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = MiniGeminiLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        if "gemma" in model_args.model_name_or_path:
            # import pdb;pdb.set_trace()
            model = MiniGeminiGemmaForCausalLM.from_pretrained(   ### liuwei
            # model = MiniGeminiLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = MiniGeminiLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif "gemma" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        # fix bugs after special token with use_fast=True
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif "gemma" in model_args.version:
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
        data_args.video_processor = copy.deepcopy(vision_tower.image_processor)
        data_args.is_multimodal = True

        model.config.image_grid = data_args.image_grid
        model.config.image_global = data_args.image_global
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        if model_args.optimize_vision_tower:
            print('Optimize last 1/2 layers in vision tower')
            total_num = len(vision_tower.vision_tower.vision_model.encoder.layers)
            for _idx in range(total_num // 2, total_num):
                vision_tower.vision_tower.vision_model.encoder.layers[_idx].requires_grad_(True)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    else:
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if model_args.freeze_backbone:
        model.requires_grad_(False)
    else:
        model.requires_grad_(True)

    if model_args.reload_embedder:
        if model_args.tune_mmprojector:
            print('frozen LLM  and reload [quantizer, attenprojection]')
            model.get_model().initialize_embedder(
                quantizer_pth='unitok_full_1ep_quantizer.pth',
                attention_pth='unitok_full_1ep_attenprojection.pth',
                mm_projecter_pth=None
            )
            for p in model.get_model().multi_embedder.quantizer.parameters():
                p.requires_grad = False
            for p in model.get_model().multi_embedder.attn_projection.parameters():
                p.requires_grad = False
            for p in model.get_model().multi_embedder.mm_projector.parameters():
                p.requires_grad = True
            model.get_model().multi_embedder.quantizer.eval()
            model.get_model().multi_embedder.attn_projection.eval()
        else:
            print('reload [quantizer, attenprojection,  prealigned_mm_projecter_pth]')
            model.get_model().initialize_embedder(
                quantizer_pth='unitok_full_1ep_quantizer.pth',
                attention_pth='unitok_full_1ep_attenprojection.pth',
                mm_projecter_pth=model_args.pretrained_mmprojector
            )  # load pretrained projector
            for p in model.get_model().multi_embedder.quantizer.parameters():
                p.requires_grad = False
            for p in model.get_model().multi_embedder.attn_projection.parameters():
                p.requires_grad = True
            for p in model.get_model().multi_embedder.mm_projector.parameters():
                p.requires_grad = True
            model.get_model().multi_embedder.quantizer.eval()
    else:
        for p in model.get_model().multi_embedder.quantizer.parameters():
            p.requires_grad = False
        for p in model.get_model().multi_embedder.attn_projection.parameters():
            p.requires_grad = True
        for p in model.get_model().multi_embedder.mm_projector.parameters():
            p.requires_grad = True

    if model_args.vision_tower_aux is not None:
        vision_tower_aux = model.get_vision_tower_aux()
        vision_tower_aux.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        assert data_args.image_processor.image_mean == vision_tower_aux.config['preprocess_cfg']['mean'] \
               and data_args.image_processor.image_std == vision_tower_aux.config['preprocess_cfg']['std'], \
            'image processor should be the same'

        if model_args.optimize_vision_tower_aux:
            print('Optimize last layer of each block in vision tower aux')
            for _idx in range(len(vision_tower_aux.vision_stages)):
                vision_tower_aux.vision_stages[_idx].blocks[-1].requires_grad_(True)

        data_args.image_size_raw = data_args.image_processor.crop_size.copy()
        model_args.image_size_aux = data_args.image_size_aux
        data_args.image_processor.crop_size['height'] = data_args.image_size_aux
        data_args.image_processor.crop_size['width'] = data_args.image_size_aux
        data_args.image_processor.size['shortest_edge'] = data_args.image_size_aux

        model.get_model().initialize_uni_modules(model_args)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # import pdb;pdb.set_trace()
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    global MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update([('mini_gemini_gemma', 'MiniGeminiGemmaForCausalLM')])

    # import pdb;pdb.set_trace()

    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    # import pdb;pdb.set_trace()
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
