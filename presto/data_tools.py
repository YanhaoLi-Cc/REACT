from typing import Dict, List, Any, Union, Optional
from collections import Counter
from functools import cache
import contextlib
import tempfile
import shutil
import random
import subprocess
import json
import re
import io
import os
import logging

import torch
import requests
import transformers
import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image

from presto.constants import IGNORE_INDEX
from presto.modalities.base_modality import Modality

def encode_chat(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List[Modality],
) -> Dict:
    messages = list(item["messages"])
    chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

    token_to_modality = {m.token: m for m in modalities}
    modality_token_counts = Counter()
    modality_instance_counts = Counter()
    instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
    pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

    chat_part = re.split(instruct_pattern, chat_as_string)
    input_ids = []
    labels = []
    
    data_dict = dict()
    for m in modalities:
        data_dict[m.name] = m.preprocess_rows([item])[0]

    for part in chat_part:
        if "[INST]" in part:
            is_instruction = True
        else:
            is_instruction = False
        for subpart in re.split(pattern, part):
            if not subpart:
                continue
            if subpart in token_to_modality:
                assert (
                    is_instruction
                ), "There should be no modality tokens outside of instructions"
                m = token_to_modality[subpart]
                m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
                modality_instance_counts[m.name] += 1
                modality_token_counts[m.name] += m_token_width
                input_ids.extend([m.token_idx] * m_token_width)
                labels.extend([IGNORE_INDEX] * m_token_width)
            elif is_instruction:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    data_dict.update({"input_ids": input_ids, "labels": labels})
    return data_dict


def encode_dpo_chat(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer, 
    modalities: List[Modality],
) -> Dict:
    # 获取词汇表大小并添加日志
    vocab_size = tokenizer.vocab_size
    logging.info(f"Tokenizer vocab size: {vocab_size}")

    # 辅助函数：确保token id在有效范围内
    def clamp_token_id(token_id):
        return min(max(0, token_id), vocab_size - 1)

    def process_sequence(messages):
        chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)
        
        token_to_modality = {m.token: m for m in modalities}
        modality_instance_counts = Counter()
        instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
        pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

        chat_part = re.split(instruct_pattern, chat_as_string)
        input_ids = []
        labels = []

        for part in chat_part:
            if "[INST]" in part:
                is_instruction = True
            else:
                is_instruction = False
            
            for subpart in re.split(pattern, part):
                if not subpart:
                    continue
                    
                if subpart in token_to_modality:
                    assert is_instruction, "There should be no modality tokens outside of instructions"
                    m = token_to_modality[subpart]
                    m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
                    modality_instance_counts[m.name] += 1
                    # 确保模态token_idx在范围内
                    safe_token_idx = clamp_token_id(m.token_idx)
                    input_ids.extend([safe_token_idx] * m_token_width)
                    labels.extend([IGNORE_INDEX] * m_token_width)
                elif is_instruction:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    # 确保常规token在范围内
                    part_ids = [clamp_token_id(tid) for tid in part_ids]
                    input_ids.extend(part_ids)
                    labels.extend([IGNORE_INDEX] * len(part_ids))
                else:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    # 确保常规token在范围内
                    part_ids = [clamp_token_id(tid) for tid in part_ids]
                    input_ids.extend(part_ids)
                    labels.extend(part_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # 添加调试信息
        logging.debug(f"Max token id in sequence: {input_ids.max().item()}")
        logging.debug(f"Min token id in sequence: {input_ids.min().item()}")
        logging.debug(f"Sequence length: {len(input_ids)}")
        
        return input_ids, labels, attention_mask

    # 数据验证
    assert "chosen_messages" in item and "rejected_messages" in item, "Missing messages in item"
    assert isinstance(item["chosen_messages"], (list, tuple)), "chosen_messages must be a list"
    assert isinstance(item["rejected_messages"], (list, tuple)), "rejected_messages must be a list"

    # 处理模态数据
    data_dict = dict()
    for m in modalities:
        data_dict[m.name] = m.preprocess_rows([item])[0]

    # 处理chosen和rejected序列
    chosen_input_ids, chosen_labels, chosen_attention_mask = process_sequence(item["chosen_messages"])
    rejected_input_ids, rejected_labels, rejected_attention_mask = process_sequence(item["rejected_messages"])

    # 截断处理
    max_length = tokenizer.model_max_length
    if max_length:
        if len(chosen_input_ids) > max_length:
            chosen_input_ids = chosen_input_ids[:max_length]
            chosen_labels = chosen_labels[:max_length]
            chosen_attention_mask = chosen_attention_mask[:max_length]
        if len(rejected_input_ids) > max_length:
            rejected_input_ids = rejected_input_ids[:max_length]
            rejected_labels = rejected_labels[:max_length]
            rejected_attention_mask = rejected_attention_mask[:max_length]

    # 最终验证
    assert chosen_input_ids.max() < vocab_size, f"Chosen input_ids contains invalid token: {chosen_input_ids.max()}"
    assert rejected_input_ids.max() < vocab_size, f"Rejected input_ids contains invalid token: {rejected_input_ids.max()}"

    # 更新数据字典
    data_dict.update({
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask.float(),
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask.float(),
    })

    return data_dict
 
def encode_chat_phi2(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List['Modality'],  # Replace 'Modality' with your actual Modality class
) -> Dict:
    messages = list(item["messages"])
    chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)
    
    token_to_modality = {m.token: m for m in modalities}
    modality_token_counts = Counter()
    modality_instance_counts = Counter()
    
    # Define patterns for Phi-2
    im_start_pattern = r"<\|im_start\|>(user|assistant)"
    im_end_pattern = r"<\|im_end\|>"
    modality_pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
    
    # Split the chat into role-message pairs
    splits = re.split(f"({im_start_pattern})", chat_as_string)
    
    # Initialize input_ids and labels
    input_ids = []
    labels = []
    
    data_dict = {m.name: m.preprocess_rows([item])[0] for m in modalities}
    
    current_role = None
    for split in splits:
        if re.match(im_start_pattern, split):
            current_role = re.match(im_start_pattern, split).group(1)
            continue  # Move to the next split which should be the message
        elif re.match(im_end_pattern, split):
            current_role = None
            continue  # Move to the next split
        elif current_role:
            # Process the message based on the current role
            message = split.strip()
            # Optionally, you can prepend role-specific tokens if needed
            # For Phi-2, roles are already specified, so you might not need to add extra tokens
            # Handle modalities within the message if any
            for subpart in re.split(modality_pattern, message):
                if not subpart:
                    continue
                if subpart in token_to_modality:
                    m = token_to_modality[subpart]
                    modality_instance = modality_instance_counts[m.name]
                    m_token_width = data_dict[m.name][modality_instance][0].shape[0]
                    modality_instance_counts[m.name] += 1
                    modality_token_counts[m.name] += m_token_width
                    input_ids.extend([m.token_idx] * m_token_width)
                    labels.extend([IGNORE_INDEX] * m_token_width)
                else:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    input_ids.extend(part_ids)
                    if current_role == "assistant":
                        labels.extend([IGNORE_INDEX] * len(part_ids))  # Typically, labels are ignored for assistant outputs
                    else:
                        labels.extend(part_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Add modality counts if needed
    for m in modalities:
        data_dict[m.name] = m.preprocess_rows([item])[0]
    
    data_dict.update({"input_ids": input_ids, "labels": labels})
    return data_dict
    
def encode_interleaved_data(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List[Modality],
):  
    token_to_modality = {m.token: m for m in modalities}
    pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
    input_ids = []
    labels = []
    data_dict = dict()
    modality_instance_counts = Counter()
    modality_token_counts = Counter()
    
    for m in modalities:
        # ensure the item has key like "smiles" or "selfies"
        data_dict[m.name] = m.preprocess_rows([item])[0]
        
    # convert the multi-turns "messages" into a single string
    if "messages" in item and "text" not in item:
        text_str = ""
        for turn in item["messages"]:
            text_str += turn["content"]
        item["text"] = text_str
    for subpart in re.split(pattern, item["text"]):
        if not subpart:
            continue
        if subpart in token_to_modality:
            m = token_to_modality[subpart]
            m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
            modality_instance_counts[m.name] += 1
            modality_token_counts[m.name] += m_token_width
            input_ids.extend([m.token_idx] * m_token_width)
            labels.extend([IGNORE_INDEX] * m_token_width)
        else:
            part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
            input_ids.extend(part_ids)
            labels.extend(part_ids)
    
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    data_dict.update({"input_ids": input_ids, "labels": labels})
    return data_dict



def parse_chat_output(output: str, style: str = "base") -> Dict:
    if style == "base":
        pattern_thoughts = r"Thoughts:(?:\n| )([\s\S]*?)\n"
        pattern_output = r"Output:(?:\n| )([\s\S]*)"
        thoughts = re.search(pattern_thoughts, output)
        if thoughts:
            thoughts = thoughts.group(1).strip()
        else:
            thoughts = None
        output = re.search(pattern_output, output).group(1).strip()
        return {"output": output, "thoughts": thoughts}
    elif style == "classification":
        # extract int from output
        thoughts = None # temporarily set to None
        output = int(re.search(r"\d+", output).group())
        return {"output": output, "thoughts": thoughts}
    elif style == "regression":
        # extract float from output
        thoughts = None # temporarily set to None
        try:
            output = float(re.search(r"\d+\.\d+", output).group())
        except:
            output = float(re.search(r"\d+", output).group())
        return {"output": output, "thoughts": thoughts}
    else:
        raise ValueError(f"Invalid style: {style}")
        

@contextlib.contextmanager
def with_local_files(fn_or_urls: List[Any]):
    local_fns = []
    fps = []
    for fn_or_url in fn_or_urls:
        if isinstance(fn_or_url, Image.Image):
            fp = tempfile.NamedTemporaryFile(suffix=".png", mode="wb")
            fn_or_url.convert("RGB").save(fp)
            fps.append(fp)
            local_fns.append(fp.name)
        elif fn_or_url.startswith("http://") or fn_or_url.startswith("https://"):
            suffix = os.path.splitext(fn_or_url)[-1]
            with requests.get(fn_or_url, stream=True) as r:
                fp = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb")
                shutil.copyfileobj(r.raw, fp)
                fps.append(fp)
                local_fns.append(fp.name)
        else:
            local_fns.append(fn_or_url)
    try:
        yield local_fns
    finally:
        for fp in fps:
            fp.close()


@cache
def _get_dataset(dataset_args: str) -> Dataset:
    return load_dataset(**json.loads(dataset_args))


def get_dataset_cached(dataset_args: Dict) -> Dataset:
    return _get_dataset(json.dumps(dataset_args))