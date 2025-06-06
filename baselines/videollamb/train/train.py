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

import os
import re
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, VIDEO_TOKEN_INDEX, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_x_token, process_anyres_image
from llava.vid_utils import load_video

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    X: Optional[List[str]] = field(default=None)
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    freeze_image_tower: bool = field(default=True)
    freeze_video_tower: bool = field(default=True)
    num_frames: Optional[int] = field(default=8)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    merge_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_x_start_end: bool = field(default=False)
    mm_use_x_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    x_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: str = '[[336, 672], [336, 1008], [336, 1344], [336, 1680], [336, 2016], [336, 2352], [336, 2688], [336, 3024], [336, 3360], [336, 3696], [336, 4032], [336, 4368], [336, 4704], [336, 5040], [336, 5376], [336, 5712], [336, 6048], [336, 6384], [336, 6720], [336, 7056], [336, 7392], [336, 7728], [336, 8064], [336, 8400], [336, 8736], [336, 9072], [336, 9408], [336, 9744], [336, 10080], [336, 10416], [336, 10752], [336, 11088], [336, 11424], [336, 11760], [336, 12096], [336, 12432], [336, 12768], [336, 13104], [336, 13440], [336, 13776], [336, 14112], [336, 14448], [336, 14784], [336, 15120], [336, 15456], [336, 15792], [336, 16128], [336, 16464], [672, 336], [672, 672], [672, 1008], [672, 1344], [672, 1680], [672, 2016], [672, 2352], [672, 2688], [672, 3024], [672, 3360], [672, 3696], [672, 4032], [672, 4368], [672, 4704], [672, 5040], [672, 5376], [672, 5712], [672, 6048], [672, 6384], [672, 6720], [672, 7056], [672, 7392], [672, 7728], [672, 8064], [1008, 336], [1008, 672], [1008, 1008], [1008, 1344], [1008, 1680], [1008, 2016], [1008, 2352], [1008, 2688], [1008, 3024], [1008, 3360], [1008, 3696], [1008, 4032], [1008, 4368], [1008, 4704], [1008, 5040], [1008, 5376], [1344, 336], [1344, 672], [1344, 1008], [1344, 1344], [1344, 1680], [1344, 2016], [1344, 2352], [1344, 2688], [1344, 3024], [1344, 3360], [1344, 3696], [1344, 4032], [1680, 336], [1680, 672], [1680, 1008], [1680, 1344], [1680, 1680], [1680, 2016], [1680, 2352], [1680, 2688], [1680, 3024], [2016, 336], [2016, 672], [2016, 1008], [2016, 1344], [2016, 1680], [2016, 2016], [2016, 2352], [2016, 2688], [2352, 336], [2352, 672], [2352, 1008], [2352, 1344], [2352, 1680], [2352, 2016], [2352, 2352], [2688, 336], [2688, 672], [2688, 1008], [2688, 1344], [2688, 1680], [2688, 2016], [3024, 336], [3024, 672], [3024, 1008], [3024, 1344], [3024, 1680], [3360, 336], [3360, 672], [3360, 1008], [3360, 1344], [3696, 336], [3696, 672], [3696, 1008], [3696, 1344], [4032, 336], [4032, 672], [4032, 1008], [4032, 1344], [4368, 336], [4368, 672], [4368, 1008], [4704, 336], [4704, 672], [4704, 1008], [5040, 336], [5040, 672], [5040, 1008], [5376, 336], [5376, 672], [5376, 1008], [5712, 336], [5712, 672], [6048, 336], [6048, 672], [6384, 336], [6384, 672], [6720, 336], [6720, 672], [7056, 336], [7056, 672], [7392, 336], [7392, 672], [7728, 336], [7728, 672], [8064, 336], [8064, 672], [8400, 336], [8736, 336], [9072, 336], [9408, 336], [9744, 336], [10080, 336], [10416, 336], [10752, 336], [11088, 336], [11424, 336], [11760, 336], [12096, 336], [12432, 336], [12768, 336], [13104, 336], [13440, 336], [13776, 336], [14112, 336], [14448, 336], [14784, 336], [15120, 336], [15456, 336], [15792, 336], [16128, 336], [16464, 336]]'
    fps: Optional[int] = field(default=None),
    max_frames: Optional[int] = field(default=None)


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

    tokenizer_model_max_length: Optional[int] = None


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
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_x_start_end", False):
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
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
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
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    # normalize special tokens
    for source in sources:
        for sentence in source:
            for DEFAULT_TOKEN in DEFAULT_X_TOKEN.values():
                X = DEFAULT_TOKEN[1:-1]
                if DEFAULT_TOKEN in sentence["value"]:
                    sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_TOKEN + '\n' + sentence['value']
                    # sentence['value'] = DEFAULT_X_TOKEN['IMAGE'] + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, f'<{X.capitalize()}>' + DEFAULT_TOKEN + f'<{X.capitalize()}>')
            replace_token = DEFAULT_TOKEN
            if data_args.mm_use_x_start_end:
                replace_token = DEFAULT_X_START_TOKEN[X.upper()] + replace_token + DEFAULT_X_END_TOKEN[X.upper()]
            sentence["value"] = sentence["value"].replace(DEFAULT_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None,
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

    if X is not None:
        input_ids = torch.stack([tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
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

            if X is not None:
                round_len = len(tokenizer_x_token(rou, tokenizer, X_TOKEN_INDEX[X]))
                instruction_len = len(tokenizer_x_token(parts[0], tokenizer, X_TOKEN_INDEX[X])) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

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
    X: str = None,
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

    if X is not None:
        input_ids = torch.stack([tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
        # input_ids = torch.stack([tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX["IMAGE"], return_tensors='pt') for prompt in conversations], dim=0)
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

    # Mask targets: special tokens (<image>, <video>)
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
                break
            parts[0] += sep

            if X is not None:
                round_len = len(tokenizer_x_token(rou, tokenizer, X_TOKEN_INDEX[X]))
                instruction_len = len(tokenizer_x_token(parts[0], tokenizer, X_TOKEN_INDEX[X])) - 2
                # round_len = len(tokenizer_x_token(rou, tokenizer, X_TOKEN_INDEX["IMAGE"]))
                # instruction_len = len(tokenizer_x_token(parts[0], tokenizer, X_TOKEN_INDEX["IMAGE"])) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

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
    X: str = None,
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

    if X is not None:
        input_ids = torch.stack([tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
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
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if X is not None:
                round_len = len(tokenizer_x_token(rou, tokenizer, X_TOKEN_INDEX[X]))
                instruction_len = len(tokenizer_x_token(parts[0], tokenizer, X_TOKEN_INDEX[X])) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

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


def preprocess_qwen(
    sources, 
    tokenizer: transformers.PreTrainedTokenizer, 
    has_image: bool = False, 
    X: str = None,
    max_len=2048, 
    system_message: str = "You are a helpful assistant."
    ) -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # if has_image and "<image>" in sentence["value"]:
            if X is not None:
                num_im = len(re.findall(DEFAULT_X_TOKEN[X], sentence["value"]))
                # multi image can start with text first
                if num_im == 1:
                    assert sentence["value"].startswith(DEFAULT_X_TOKEN[X]), print(sentence["value"])
                matches = re.findall(DEFAULT_X_TOKEN[X], sentence["value"])
                num_image = len(matches)
                _input_id = tokenizer(role).input_ids + nl_tokens + [X_TOKEN_INDEX[X]] * num_image + nl_tokens + tokenizer(sentence["value"][len(DEFAULT_X_TOKEN[X]) * num_image :]).input_ids + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )




def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None,
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

    if X is not None:
        input_ids = torch.stack([tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
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
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i > 0:
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


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None,
) -> Dict:
    # add end signal and concatenate together
    DEFAULT_TOKEN = DEFAULT_X_TOKEN[X]
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    if X is not None:
        input_ids = [tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations]
        # input_ids = [tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX["IMAGE"], return_tensors='pt') for prompt in conversations]
    else:
        input_ids = [tokenizer(prompt).input_ids for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if X is not None:
            tokenized_len = len(tokenizer_x_token(source[0]['value'], tokenizer, X_TOKEN_INDEX[X]))
            # tokenized_len = len(tokenizer_x_token(source[0]['value'], tokenizer, X_TOKEN_INDEX["IMAGE"]))
        else:
            tokenized_len = len(tokenizer(source[0]['value']).input_ids)
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, X)
        # return preprocess_plain(sources, tokenizer, "IMAGE")
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, X)
        # return preprocess_llama_2(sources, tokenizer, "IMAGE")
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, X)
        # return preprocess_v1(sources, tokenizer, "IMAGE")
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, X)
        # return preprocess_mpt(sources, tokenizer, "IMAGE")
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, X)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=True, X=X)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X])) for prompt in prompts]
        # return [len(tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX["IMAGE"])) for prompt in prompts]

    if X is not None:
        input_ids = [tokenizer_x_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if X is not None:
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
        list_data_dict = json.load(open(data_path, "r"))
        
        # group modality
        grouped_list_data_dict = {"image":[], "video":[], "language":[]}
        for dct in list_data_dict:
            if "image" in dct:
                grouped_list_data_dict["image"].append(dct)
            elif "video" in dct:
                grouped_list_data_dict["video"].append(dct)
            else:
                grouped_list_data_dict["language"].append(dct)
        self.grouped_list_data_dict = grouped_list_data_dict

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)
    
    def __modality_len__(self, modality):
        return len(self.grouped_list_data_dict[modality])

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            modality = "language"
            # cur_len = cur_len if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else -cur_len
            # currently, each language pair with single modality
            for x in X_TOKEN_INDEX.keys():
                if x.lower() in sample:
                    # cur_len = X_TOKEN_INDEX[x]
                    modality = x.lower()
                    break # FIXME: one modality per data item
            length_list.append([modality, cur_len])
        return length_list

    def __getitem__(self, i, modality=None) -> Dict[str, torch.Tensor]:
        # BUG: same type
        try:
            X = None
            if modality is not None:
                sources = self.grouped_list_data_dict[modality][i]
            else:
                sources = self.list_data_dict[i]
                
            # # TODO: OOM: frame numbers
            # print("*"*20)
            # print(sources)
            # print("*"*20)
            # if "video" in sources and "1063538854.mp4" in sources["video"]: raise ValueError("oom")
                
            # print(sources["conversations"])
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            if 'image' in sources[0]:
                x_modality = 'image'
                # image_file = self.list_data_dict[i]['image']
                image_file = sources[0]['image']
                x_folder = self.data_args.x_folder
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(x_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                elif self.data_args.image_aspect_ratio == "anyres":
                    image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                pro_sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
                X = 'IMAGE'

            elif 'video' in sources[0]:
                x_modality = 'video'
                # video_file = self.list_data_dict[i]['video']
                video_file = sources[0]['video']
                x_folder = self.data_args.x_folder
                # # HACK: debug
                # x_folder = '/'
                processor = self.data_args.video_processor
                video = os.path.join(x_folder, video_file)
                # video = read_videos(video, 16)
                # video = processor(list(video), return_tensors='pt')['pixel_values'][0]
                
                if "LanguageBind" in self.data_args.video_tower:
                    # print("*"*20)
                    # print(self.data_args.fps)
                    # print("*"*20)
                    # video = processor(video, fps=self.data_args.fps, return_tensors='pt')['pixel_values'][0] # make_list_of_images
                    video = processor(video, return_tensors='pt')['pixel_values'][0] # make_list_of_images
                else:
                    if os.path.isdir(video):
                        video = load_video(video, video_decode_backend='frame', num_frames=self.data_args.num_frames, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                    elif video.endswith(".gif"):
                        video = load_video(video, video_decode_backend='gif', num_frames=self.data_args.num_frames, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                    else:
                        video = load_video(video, video_decode_backend='decord', num_frames=self.data_args.num_frames, fps=self.data_args.fps, max_frames=self.data_args.max_frames)
                    video = processor(video, return_tensors="pt")["pixel_values"]
                
                pro_sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args
                )
                X = 'VIDEO'


            else:
                x_modality = "language"
                pro_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess(
                pro_sources,
                self.tokenizer,
                X)
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])

            # image exist in the data
            # if 'image' in self.list_data_dict[i]:
            if 'image' in sources[0]:
                data_dict['image'] = image
            # elif 'video' in self.list_data_dict[i]:
            elif 'video' in sources[0]:
                data_dict['video'] = video
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                if hasattr(self.data_args, 'image_processor'):
                    if hasattr(self.data_args.image_processor, 'crop_size'):
                        size = self.data_args.image_processor.crop_size
                    elif hasattr(self.data_args.image_processor, 'size'):
                        size = self.data_args.image_processor.size
                    else:
                        # size = {"height": 224, "width": 224}
                        size = {"height": 336, "width": 336}
                    data_dict['image'] = torch.zeros(3, size['height'], size['width'])
                elif hasattr(self.data_args, 'video_processor'):
                    if hasattr(self.data_args.video_processor, 'crop_size'):
                        size = self.data_args.video_processor.crop_size
                    elif hasattr(self.data_args.video_processor, 'size'):
                        size = self.data_args.video_processor.size
                    else:
                        # size = {"height": 224, "width": 224}
                        size = {"height": 336, "width": 336}
                    data_dict['video'] = torch.zeros(3, 1, size['height'], size['width'])
                else:
                    raise NotImplementedError("you should specify at least one vision encoder")
            return data_dict
        except Exception as e: # FIXME
            print(f"DataLoader Error with {e}")
            return self.__getitem__(random.randint(0, self.__modality_len__(x_modality)-1), modality=x_modality)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # # input_ids = torch.nn.utils.rnn.pad_sequence(
        # #     input_ids,
        # #     batch_first=True,
        # #     padding_value=IGNORE_INDEX)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        # print(input_ids.shape)
        
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # if 'image' in instances[0]:
        #     images = [instance['image'] for instance in instances]
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images

        X, X_modalities = [], []
        for instance in instances:
            # print(instance)
            for x in DEFAULT_X_TOKEN.keys():
                if x.lower() in instance:
                    X.append(instance[x.lower()])
                    X_modalities.append(x)
        batch["X"] = X
        batch["X_modalities"] = X_modalities
        batch["X_sizes"] = [None] * len(X)

        # print(len(instances))
        # print(X_modalities)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
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
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.image_tower is not None or model_args.video_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        
        elif 'mistral' in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            
        elif 'qwen' in model_args.model_name_or_path.lower():
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        
        else:
            
            if "_rmt" in model_args.model_name_or_path:
                model = LlavaLlamaForCausalLMRMT.from_pretrained(
                    model_args.model_name_or_path.split("_rmt")[0],
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.config.X = model_args.X

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
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
    elif 'mistral' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left"
        )
    elif 'qwen' in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        if 'rmt' in model_args.model_name_or_path:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path.split('_rmt')[0],
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
        else:
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
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        if tokenizer.pad_token is None and ('llama3' in model_args.version or 'llama_3' in model_args.version):
            # print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
            # tokenizer.pad_token = tokenizer.eos_token
        else:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.image_tower is not None or model_args.video_tower is not None:

        if model_args.image_tower is not None:
            model.get_model().initialize_image_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            
            image_tower = model.get_image_tower()
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            # model.model.image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            
            
            # print("=========to method===========")
            # print(model.model.image_tower.dtype)
            # print(model.model.image_tower.image_tower.dtype)
            # print("====================")
            # # raise ValueError("debug")

            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
        
        if model_args.video_tower is not None:
            model.get_model().initialize_video_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            video_tower = model.get_video_tower()
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.video_tower = model_args.video_tower
            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        # model.config.tokenizer_model_max_length = tokenizer.model_max_length # prevent from hang out

        tokenizer_model_max_length = training_args.tokenizer_model_max_length
        model.config.tokenizer_model_max_length = tokenizer_model_max_length if tokenizer_model_max_length else tokenizer.model_max_length    

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

        model.config.mm_use_x_start_end = data_args.mm_use_x_start_end = model_args.mm_use_x_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_x_start_end = model_args.mm_use_x_start_end
        model.config.mm_use_x_patch_token = model_args.mm_use_x_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        # model.config.pad_token_id = tokenizer.pad_token_id

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

    data_args.num_frames = model_args.num_frames
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # print("=========to method before ===========")
    # print(model.dtype)
    # print(model.model.dtype)
    # print(model.model.image_tower.image_tower.dtype)
    # print("====================")
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # print("=========to method after ===========")
    # print(model.model.image_tower.image_tower.dtype)
    # print("====================")

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
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
