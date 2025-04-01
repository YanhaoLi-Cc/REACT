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
    """
    更安全的encode_chat版本，为DPO训练设计，添加更多的验证和错误处理
    """
    try:
        messages = list(item.get("messages", []))
        
        # 验证messages格式
        if not messages:
            raise ValueError("Empty messages list")
        
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Message is not a dictionary: {type(msg)}")
            if "role" not in msg:
                raise ValueError("Message missing 'role' field")
            if "content" not in msg:
                raise ValueError("Message missing 'content' field")
        
        try:
            # 尝试应用聊天模板，如果失败则使用简单的格式
            chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            # 如果应用模板失败，使用简单的拼接
            logging.warning(f"Failed to apply chat template: {str(e)}, using simple format")
            chat_as_string = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        token_to_modality = {m.token: m for m in modalities}
        modality_token_counts = Counter()
        modality_instance_counts = Counter()
        
        # 尝试使用模板的正则表达式
        try:
            instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
            pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
            
            chat_part = re.split(instruct_pattern, chat_as_string)
        except Exception as e:
            # 如果正则匹配失败，直接使用整个字符串
            logging.warning(f"Failed to split chat with regex: {str(e)}")
            chat_part = [chat_as_string]
        
        input_ids = []
        labels = []
        
        # 安全地预处理模态数据
        data_dict = dict()
        for m in modalities:
            try:
                data_dict[m.name] = m.preprocess_rows([item])[0]
            except Exception as e:
                logging.warning(f"Failed to preprocess modality {m.name}: {str(e)}")
                data_dict[m.name] = []
        
        # 处理聊天部分
        for part in chat_part:
            is_instruction = "[INST]" in part if isinstance(part, str) else False
            
            try:
                if isinstance(part, str):
                    subparts = re.split(pattern, part)
                else:
                    subparts = [str(part)]
                    
                for subpart in subparts:
                    if not subpart:
                        continue
                        
                    if subpart in token_to_modality:
                        m = token_to_modality[subpart]
                        
                        # 检查是否有足够的模态实例
                        if (m.name not in data_dict or 
                            len(data_dict[m.name]) <= modality_instance_counts[m.name]):
                            # 如果没有，跳过这个模态
                            logging.warning(f"Missing modality instance for {m.name}")
                            continue
                            
                        try:
                            m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
                            modality_instance_counts[m.name] += 1
                            modality_token_counts[m.name] += m_token_width
                            input_ids.extend([m.token_idx] * m_token_width)
                            labels.extend([IGNORE_INDEX] * m_token_width)
                        except Exception as e:
                            logging.warning(f"Failed to process modality token: {str(e)}")
                            # 使用一个默认值
                            input_ids.extend([m.token_idx] * 10)
                            labels.extend([IGNORE_INDEX] * 10)
                    elif is_instruction:
                        part_ids = tokenizer(str(subpart), add_special_tokens=False).input_ids
                        input_ids.extend(part_ids)
                        labels.extend([IGNORE_INDEX] * len(part_ids))
                    else:
                        part_ids = tokenizer(str(subpart), add_special_tokens=False).input_ids
                        input_ids.extend(part_ids)
                        labels.extend(part_ids)
            except Exception as e:
                logging.warning(f"Failed to process chat part: {str(e)}")
                # 对于失败的部分，添加一些占位符token
                placeholder_ids = tokenizer("Error processing this part.", add_special_tokens=False).input_ids
                input_ids.extend(placeholder_ids)
                if is_instruction:
                    labels.extend([IGNORE_INDEX] * len(placeholder_ids))
                else:
                    labels.extend(placeholder_ids)
        
        # 确保input_ids和labels非空
        if not input_ids:
            input_ids = [tokenizer.pad_token_id]
        if not labels:
            labels = [IGNORE_INDEX]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        # 添加模态数据
        for m in modalities:
            if m.name in data_dict:
                result[m.name] = data_dict[m.name]
        
        return result
    except Exception as e:
        # 如果发生任何错误，返回一个最小的有效数据
        logging.error(f"Unhandled error in encode_dpo_chat: {str(e)}")
        # 确保返回一个有效的attention_mask
        return {
            "input_ids": torch.tensor([tokenizer.pad_token_id], dtype=torch.long),
            "labels": torch.tensor([IGNORE_INDEX], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
        }

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