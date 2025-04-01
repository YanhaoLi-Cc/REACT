# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import List, Dict, Sequence, Optional, Union
from dataclasses import dataclass, field
import logging
import os
import json

import transformers
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import  ConcatDataset, Subset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
# from trl.trainer import DPODataCollatorWithPadding

from torch.nn.utils.rnn import pad_sequence

from presto.modalities.base_modality import Modality
from presto.constants import IGNORE_INDEX
from presto.data_tools import encode_chat, encode_dpo_chat, encode_interleaved_data

_DATASETS = {}
_MIXTURES = {}
DATASET_BASE_DIR="/home/liyanhao/chemllm/REACT/datasets/"

@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data. (Will be deprecated in future versions)"}
    )


class DatasetType(Enum):
    CHAT = "chat"
    INTERLEAVED = "interleaved"


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default=DatasetType.CHAT)
    train_path: str = field(
        default='', metadata={"help": "Path to the training data."}
    )
    eval_path: str = field(
        default='', metadata={"help": "Path to the evaluation data."}
    )
    test_path: str = field(
        default='', metadata={"help": "Path to the test data."}
    )
    repo_id: str = field(
        default='', metadata={"help": "Hugging Face dataset repository ID."}
    )

def _register_dataset(name, type, train_path='', eval_path='', test_path='', repo_id=''):
    dataset = Dataset(dataset_name=name, dataset_type=type, train_path=train_path, eval_path=eval_path, test_path=test_path, repo_id=repo_id)
    _DATASETS[name] = dataset


def _register_mixture(mixture_name, dataset_names: Dict[str, float]):
    fracs = dataset_names.values()
    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")
    if all(name in _DATASETS for name in dataset_names):
        _MIXTURES[mixture_name] = [(_DATASETS[name], frac) for name, frac in dataset_names.items()]
    else:
        raise ValueError("One or more dataset names provided do not exist in the dataset registry.")


def _resolve_dataset(args: Dataset, split: str):
    split_path = getattr(args, f"{split}_path", None)
    print(f"Checking split_path: {split_path}")  # 调试信息
    
    if split_path and os.path.exists(split_path):
        # 检查是否是JSON文件
        if split_path.endswith('.json'):
            print(f"Loading JSON file from: {split_path}")  # 调试信息
            try:
                with open(split_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 将JSON数据转换为HuggingFace数据集格式
                return HFDataset.from_list(data)
            except Exception as e:
                print(f"Error loading JSON: {str(e)}")  # 调试信息
                raise ValueError(f"Error loading JSON file {split_path}: {str(e)}")
        return load_from_disk(split_path)
    
    if args.repo_id:
        try:
            dataset = load_dataset(args.repo_id)
            if split in dataset.keys():
                return dataset[split]
            elif split == "eval" and "validation" in dataset.keys():
                return dataset["validation"]
            elif split == "eval" and "valid" in dataset.keys():
                return dataset["valid"]
            elif split == "eval" and "val" in dataset.keys():
                return dataset["val"]
            else:
                return dataset["train"]
        except:
            raise ValueError(f"Dataset {args.dataset_name} not found in the Hugging Face dataset hub.")
    
    raise ValueError(f"Dataset {args.dataset_name} not found.")

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                training_args: transformers.TrainingArguments,
                                modalities: List[Modality],
                                ) -> Dict:

    if training_args.training_mode == "sft":
        if training_args.dataset_name is not None:
            print(f"Using dataset: {training_args.dataset_name}")
            assert training_args.dataset_name in _DATASETS, f"Dataset {training_args.dataset_name} not found in registry."
            dataset = _DATASETS[training_args.dataset_name]
            train_dataset = _resolve_dataset(dataset, split="train")
            eval_dataset = _resolve_dataset(dataset, split="eval") if dataset.eval_path or dataset.repo_id else None
        elif training_args.data_mixture is not None:
            print(f"Using dataset mixture: {training_args.data_mixture}")
            assert training_args.data_mixture in _MIXTURES, f"Dataset mixture {training_args.data_mixture} not found in registry."
            mixture = _MIXTURES[training_args.data_mixture]
            train_datasets = []
            eval_datasets = []
            for data_args, frac in mixture:
                dataset_cls = _CLS_MAPPING[data_args.dataset_type]
                train_dataset = dataset_cls(tokenizer=tokenizer, modalities=modalities, data_args=data_args, split="train")
                train_subset = Subset(train_dataset, range(int(len(train_dataset)*frac)))
                train_datasets.append(train_subset)
                if training_args.eval_path:
                    eval_datasets.append(dataset_cls(tokenizer=tokenizer, modalities=modalities, data_args=data_args, split="eval"))
            train_dataset = LMMConcatDataset(train_datasets)
            if training_args.eval_path:
                eval_dataset = LMMConcatDataset(eval_datasets)
            else:
                eval_dataset = None
        else:
            raise ValueError("No dataset or dataset mixture specified.")
        data_collator = DataCollatorForSupervisedLMMDataset(tokenizer=tokenizer, modalities=modalities)
        print(f"Train dataset length: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"Eval dataset length: {len(eval_dataset)}")
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    elif training_args.training_mode == "dpo":
        if training_args.dataset_name is not None:
            print(f"Using dataset for DPO: {training_args.dataset_name}")
            assert training_args.dataset_name in _DATASETS, f"Dataset {training_args.dataset_name} not found in registry."
            dataset = _DATASETS[training_args.dataset_name]
            train_dataset = DPODataset(
                tokenizer=tokenizer,
                modalities=modalities,
                data_args=dataset,
                split="train"
            )
            eval_dataset = DPODataset(
                tokenizer=tokenizer,
                modalities=modalities,
                data_args=dataset,
                split="eval"
            ) if dataset.eval_path or dataset.repo_id else None
            
        elif training_args.data_mixture is not None:
            print(f"Using dataset mixture for DPO: {training_args.data_mixture}")
            assert training_args.data_mixture in _MIXTURES, f"Dataset mixture {training_args.data_mixture} not found in registry."
            mixture = _MIXTURES[training_args.data_mixture]
            
            train_datasets = []
            eval_datasets = []
            for data_args, frac in mixture:
                train_dataset = DPODataset(
                    tokenizer=tokenizer,
                    modalities=modalities, 
                    data_args=data_args,
                    split="train"
                )
                train_subset = Subset(train_dataset, range(int(len(train_dataset)*frac)))
                train_datasets.append(train_subset)
                
                if data_args.eval_path or data_args.repo_id:
                    eval_datasets.append(
                        DPODataset(
                            tokenizer=tokenizer,
                            modalities=modalities,
                            data_args=data_args,
                            split="eval"
                        )
                    )
                    
            train_dataset = LMMConcatDataset(train_datasets)
            eval_dataset = LMMConcatDataset(eval_datasets) if eval_datasets else None
        
        else:
            raise ValueError("No dataset or dataset mixture specified for DPO training.")
            
        data_collator = DataCollatorForDPODataset(
            tokenizer=tokenizer,
            modalities=modalities
        )
        
        print(f"DPO train dataset length: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"DPO eval dataset length: {len(eval_dataset)}")
            
        return dict(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
    else:
        raise ValueError(f"Unknown training mode: {training_args.training_mode}")


@dataclass 
class DataCollatorForDPODataset:
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]
    skip_modality_processing: bool = True  # 添加标志跳过模态处理
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 收集chosen和rejected的input_ids和labels
        chosen_input_ids = [instance["chosen_input_ids"] for instance in instances]
        chosen_labels = [instance["chosen_labels"] for instance in instances]
        
        rejected_input_ids = [instance["rejected_input_ids"] for instance in instances]
        rejected_labels = [instance["rejected_labels"] for instance in instances]
        
        # 确保所有实例都有attention_mask
        chosen_attention_mask = []
        rejected_attention_mask = []
        
        for instance in instances:
            if "chosen_attention_mask" in instance:
                chosen_attention_mask.append(instance["chosen_attention_mask"])
            else:
                # 如果没有attention_mask，创建一个全1的mask
                chosen_attention_mask.append(torch.ones_like(instance["chosen_input_ids"]))
                
            if "rejected_attention_mask" in instance:
                rejected_attention_mask.append(instance["rejected_attention_mask"])
            else:
                # 如果没有attention_mask，创建一个全1的mask
                rejected_attention_mask.append(torch.ones_like(instance["rejected_input_ids"]))

        # 计算最大长度，确保chosen和rejected具有相同的长度
        chosen_max_length = max([len(ids) for ids in chosen_input_ids])
        rejected_max_length = max([len(ids) for ids in rejected_input_ids])
        max_length = max(chosen_max_length, rejected_max_length)
        
        # Padding处理到统一长度
        padded_chosen_input_ids = []
        padded_chosen_labels = []
        padded_chosen_attention_mask = []
        padded_rejected_input_ids = []
        padded_rejected_labels = []
        padded_rejected_attention_mask = []
        
        for i in range(len(instances)):
            # Padding chosen
            if len(chosen_input_ids[i]) < max_length:
                pad_length = max_length - len(chosen_input_ids[i])
                padded_chosen_input_ids.append(torch.cat([
                    chosen_input_ids[i], 
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=chosen_input_ids[i].dtype, device=chosen_input_ids[i].device)
                ]))
                padded_chosen_labels.append(torch.cat([
                    chosen_labels[i], 
                    torch.full((pad_length,), IGNORE_INDEX, dtype=chosen_labels[i].dtype, device=chosen_labels[i].device)
                ]))
                padded_chosen_attention_mask.append(torch.cat([
                    chosen_attention_mask[i], 
                    torch.zeros(pad_length, dtype=chosen_attention_mask[i].dtype, device=chosen_attention_mask[i].device)
                ]))
            else:
                padded_chosen_input_ids.append(chosen_input_ids[i][:max_length])
                padded_chosen_labels.append(chosen_labels[i][:max_length])
                padded_chosen_attention_mask.append(chosen_attention_mask[i][:max_length])
            
            # Padding rejected
            if len(rejected_input_ids[i]) < max_length:
                pad_length = max_length - len(rejected_input_ids[i])
                padded_rejected_input_ids.append(torch.cat([
                    rejected_input_ids[i], 
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=rejected_input_ids[i].dtype, device=rejected_input_ids[i].device)
                ]))
                padded_rejected_labels.append(torch.cat([
                    rejected_labels[i], 
                    torch.full((pad_length,), IGNORE_INDEX, dtype=rejected_labels[i].dtype, device=rejected_labels[i].device)
                ]))
                padded_rejected_attention_mask.append(torch.cat([
                    rejected_attention_mask[i], 
                    torch.zeros(pad_length, dtype=rejected_attention_mask[i].dtype, device=rejected_attention_mask[i].device)
                ]))
            else:
                padded_rejected_input_ids.append(rejected_input_ids[i][:max_length])
                padded_rejected_labels.append(rejected_labels[i][:max_length])
                padded_rejected_attention_mask.append(rejected_attention_mask[i][:max_length])
        
        # 转换为batch张量
        chosen_input_ids_batch = torch.stack(padded_chosen_input_ids)
        chosen_labels_batch = torch.stack(padded_chosen_labels)
        chosen_attention_mask_batch = torch.stack(padded_chosen_attention_mask)
        
        rejected_input_ids_batch = torch.stack(padded_rejected_input_ids)
        rejected_labels_batch = torch.stack(padded_rejected_labels)
        rejected_attention_mask_batch = torch.stack(padded_rejected_attention_mask)
        
        # 截断处理
        max_model_length = self.tokenizer.model_max_length
        if max_model_length and max_length > max_model_length:
            chosen_input_ids_batch = chosen_input_ids_batch[:, :max_model_length]
            chosen_labels_batch = chosen_labels_batch[:, :max_model_length]
            chosen_attention_mask_batch = chosen_attention_mask_batch[:, :max_model_length]
            rejected_input_ids_batch = rejected_input_ids_batch[:, :max_model_length]
            rejected_labels_batch = rejected_labels_batch[:, :max_model_length]
            rejected_attention_mask_batch = rejected_attention_mask_batch[:, :max_model_length]

        batch = {
            "input_ids": torch.cat([chosen_input_ids_batch, rejected_input_ids_batch], dim=0),
            "labels": torch.cat([chosen_labels_batch, rejected_labels_batch], dim=0),
            "attention_mask": torch.cat([chosen_attention_mask_batch, rejected_attention_mask_batch], dim=0),
        }

        # 添加modality相关数据（仅当不跳过模态处理时）
        if not self.skip_modality_processing:
            for m in self.modalities:
                chosen_key = f"chosen_{m.name}"
                rejected_key = f"rejected_{m.name}"
                if chosen_key in instances[0] and rejected_key in instances[0]:
                    batch[m.name] = (
                        [instance[chosen_key] for instance in instances] + 
                        [instance[rejected_key] for instance in instances]
                    )

        return batch

    def _validate_batch(self, batch):
        """验证batch数据的有效性"""
        # 检查shape一致性
        chosen_shape = batch["chosen_input_ids"].shape
        rejected_shape = batch["rejected_input_ids"].shape
        assert chosen_shape == rejected_shape, \
            f"Shape mismatch: chosen {chosen_shape} vs rejected {rejected_shape}"
        
        # 检查attention mask的维度和类型
        assert batch["chosen_attention_mask"].shape == chosen_shape, \
            "Attention mask shape mismatch"
        assert batch["rejected_attention_mask"].shape == rejected_shape, \
            "Attention mask shape mismatch"
        
        # 检查input_ids的有效性
        assert (batch["chosen_input_ids"] >= 0).all(), "Invalid input ids found"
        assert (batch["rejected_input_ids"] >= 0).all(), "Invalid input ids found"

        # 检查labels的有效性
        valid_chosen_labels = (batch["chosen_labels"] == IGNORE_INDEX) | (batch["chosen_labels"] >= 0)
        valid_rejected_labels = (batch["rejected_labels"] == IGNORE_INDEX) | (batch["rejected_labels"] >= 0)
        assert valid_chosen_labels.all(), "Invalid chosen labels found"
        assert valid_rejected_labels.all(), "Invalid rejected labels found"

    
class LMMDataset(TorchDataset):
    def __init__(
        self,
        data_args: Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
        split: str
    ):
        super(LMMDataset, self).__init__()

        self.dataset = _resolve_dataset(data_args, split)
        self.tokenizer = tokenizer
        self.modalities = modalities

    def __len__(self):
        return len(self.dataset)

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            return encode_chat(item, self.tokenizer, self.modalities)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)


class LMMInterleavedDataset(LMMDataset):
    r"""
    Interleaved dataset for LMM pretraining. Each sample is a naive concatenation of multiple modality tokens
    and the surrounding text (Not Chat Format). The modality tokens are interleaved with the text tokens.
    """
    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            return encode_interleaved_data(item, self.tokenizer, self.modalities)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)

class DPODataset(LMMDataset):
    """Dataset for Direct Preference Optimization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_dataset()
        self.skip_modality_processing = True  # 添加标志跳过模态处理
        
    def _validate_dataset(self):
        """验证数据集格式是否符合DPO要求"""
        required_fields = ["chosen_messages", "rejected_messages"]
        for i, item in enumerate(self.dataset):
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                raise ValueError(f"Item at index {i} is missing required fields: {missing_fields}")
            if not isinstance(item["chosen_messages"], list) or not isinstance(item["rejected_messages"], list):
                raise ValueError(f"Item at index {i} contains non-list values for messages fields")
                
    def __getitem__(self, i, retry_count=0) -> Dict:
        try:
            item = self.dataset[i]
            
            # 分别为chosen和rejected创建对话数据
            chosen_item = {
                "messages": item["chosen_messages"],
                "molecules": item.get("molecules", {})
            }
            rejected_item = {
                "messages": item["rejected_messages"],
                "molecules": item.get("molecules", {})
            }
            
            if self.skip_modality_processing:
                # DPO模式：直接替换分子标记为SMILES字符串
                chosen_encoded = self._process_text_only(chosen_item)
                rejected_encoded = self._process_text_only(rejected_item)
            else:
                # 使用原有的多模态数据处理 (备用)
                chosen_encoded = encode_dpo_chat(chosen_item, self.tokenizer, self.modalities)
                rejected_encoded = encode_dpo_chat(rejected_item, self.tokenizer, self.modalities)
            
            # 构建DPO所需的格式
            result = {
                "chosen_input_ids": chosen_encoded["input_ids"],
                "chosen_labels": chosen_encoded["labels"],
                "chosen_attention_mask": chosen_encoded["attention_mask"],
                "rejected_input_ids": rejected_encoded["input_ids"],
                "rejected_labels": rejected_encoded["labels"],
                "rejected_attention_mask": rejected_encoded["attention_mask"],
            }
            
            # 如果不跳过模态处理，添加模态数据（备用）
            if not self.skip_modality_processing:
                for m in self.modalities:
                    if m.name in chosen_encoded:
                        result[f"chosen_{m.name}"] = chosen_encoded[m.name]
                    if m.name in rejected_encoded:
                        result[f"rejected_{m.name}"] = rejected_encoded[m.name]
            
            return result
            
        except Exception as e:
            # 限制重试次数，防止无限递归
            if retry_count >= 5:
                logging.error(f"Failed to get valid sample after 5 retries, returning dummy data")
                # 返回一个简单的占位数据
                return self._create_dummy_data()
                
            new_i = i + 1 if i + 1 < len(self) else 0
            logging.error(f"Error encoding DPO data: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i, retry_count + 1)
    
    def _process_text_only(self, item):
        """处理纯文本数据，将分子标记替换为SMILES字符串"""
        try:
            messages = list(item.get("messages", []))
            molecules = item.get("molecules", {}).get("smiles", [])
            
            # 创建一个新的消息列表，替换分子标记
            processed_messages = []
            for msg in messages:
                if not isinstance(msg, dict) or "content" not in msg or "role" not in msg:
                    processed_messages.append(msg)
                    continue
                
                content = msg["content"]
                # 替换<molecule_2d>标记为实际的SMILES字符串
                for i, smile in enumerate(molecules):
                    placeholder = "<molecule_2d>"
                    if placeholder in content:
                        content = content.replace(placeholder, smile, 1)
                
                processed_messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            # 使用处理后的消息列表
            chat_text = self.tokenizer.apply_chat_template(
                processed_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 编码为模型输入
            encoded = self.tokenizer(
                chat_text,
                padding=False,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors=None
            )
            
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
            # 对于自回归训练，标签与输入相同
            labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            
        except Exception as e:
            logging.error(f"Error in text-only processing: {str(e)}")
            # 返回一个最小有效数据
            return {
                "input_ids": torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
                "labels": torch.tensor([IGNORE_INDEX], dtype=torch.long),
                "attention_mask": torch.tensor([1], dtype=torch.long),
            }
            
    def _create_dummy_data(self):
        """创建一个最小的有效数据，作为失败情况的后备"""
        # 创建最小长度为1的张量
        input_ids = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        attention_mask = torch.tensor([1], dtype=torch.long)
        labels = torch.tensor([IGNORE_INDEX], dtype=torch.long)
        
        # 为chosen和rejected返回相同结构的数据，确保长度一致
        return {
            "chosen_input_ids": input_ids.clone(),
            "chosen_labels": labels.clone(),
            "chosen_attention_mask": attention_mask.clone(),
            "rejected_input_ids": input_ids.clone(),
            "rejected_labels": labels.clone(),
            "rejected_attention_mask": attention_mask.clone(),
        }

    def _validate_encoded_data(self, encoded: Dict):
        """验证编码数据的有效性"""
        # 检查必需的键
        required_keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
        for key in required_keys:
            assert key in encoded, f"Missing required key: {key}"
            assert isinstance(encoded[key], torch.Tensor), f"{key} must be a tensor"
        
        # 检查modality数据
        for m in self.modalities:
            chosen_key = f"chosen_{m.name}"
            rejected_key = f"rejected_{m.name}"
            if chosen_key in encoded:
                assert isinstance(encoded[chosen_key], list), f"{chosen_key} must be a list"
            if rejected_key in encoded:
                assert isinstance(encoded[rejected_key], list), f"{rejected_key} must be a list"


class LMMConcatDataset(ConcatDataset):
    def get_example(self) -> Dict:
        return self.datasets[0][0]


@dataclass
class DataCollatorForSupervisedLMMDataset:
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ["input_ids", "labels"]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch


_CLS_MAPPING = {DatasetType.CHAT: LMMDataset, DatasetType.INTERLEAVED: LMMInterleavedDataset}


## Register datasets
# PubChem 330K Caption Dataset
# repo_id='OpenMol/PubChem_G2S_300K_SMILES-MMPretrain'
_register_dataset(
    name="pubchem_cap",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "pubchem_caption"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/PubChem_G2S_300K_SMILES-MMPretrain'
)

# USPTO RXN Interleaved Dataset
# repo_id='OpenMol/USPTO_RXN_Interleaved'
_register_dataset(
    name="uspto_rxn",
    type=DatasetType.INTERLEAVED,
    train_path=os.path.join(DATASET_BASE_DIR, "uspto_rxn"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/USPTO_RXN_Interleaved'
)

 # Yields Regression
_register_dataset(
    name="yields_regression",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "yields_regression", "train"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/BH-SM_YR_10K-MMChat'
)
 
# Forward Prediction
_register_dataset(
    name="forward_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "forward_prediction", "train"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/MolInst_FS_125K_SMILES-MMChat'
)

# Retrosynthesis
_register_dataset(
    name="retrosynthesis",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "retrosynthesis", "train"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/MolInst_RS_125K_SMILES-MMChat'
)

# Reaction Classification
_register_dataset(
    name="reaction_classification",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reaction_classification", "train"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/MolInst_FS_125K_SMILES-MMChat'
)

# Reagent Selection
_register_dataset(
    name="reagent_selection",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reagent_selection", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/HTE_RAS_4K-MMChat",
)

# Reagent Prediction
_register_dataset(
    name="reagent_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "reagent_prediction", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/RCR_RP_57K_SMILES-MMChat",
)

# Solvent Prediction
_register_dataset(
    name="solvent_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "solvent_prediction", "train"),
    repo_id='/home/liyanhao/chemllm/REACT/datasets/RCR_SP_70K_SMILES-MMChat'
)

# Catalyst Prediction
_register_dataset(
    name="catalyst_prediction",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "catalyst_prediction", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/RCR_CP_10K_SMILES-MMChat",
)

## Name Conversion
# s2f
_register_dataset(
    name="s2f",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "s2f_mmchat_smiles", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/SMol_S2F_270K-MMChat",
)

# s2i
_register_dataset(
    name="s2i",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "s2i_mmchat_smiles", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/SMol_S2I_270K-MMChat",
)

# i2s (text-only)
_register_dataset(
    name="i2s",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "i2s_mmchat_smiles", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/SMol_I2S_270K-MMChat",
)

# i2f (text-only)
_register_dataset(
    name="i2f",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "i2f_mmchat_smiles", "train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/SMol_I2F_270K-MMChat",
)

# g2s (from PubChem)
# repo_id="OpenMol/PubChem_G2S_300K_SMILES-MMPretrain",
_register_dataset(
    name="g2s",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "g2s_mmchat_smiles",),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/PubChem_G2S_300K_SMILES-MMPretrain",
)

_register_dataset(
    name="trans",
    type=DatasetType.CHAT,
    train_path=os.path.join(DATASET_BASE_DIR, "NC", "trans_mmchat","train"),
    repo_id="/home/liyanhao/chemllm/REACT/datasets/TRANS-5K",
)

# # DPO data
# _register_dataset(
#     name="dpo_chemical",  # 数据集名称
#     type=DatasetType.CHAT,  # 使用CHAT类型，因为DPO数据也是对话格式
#     train_path=os.path.join(DATASET_BASE_DIR, "dpo_chemical", "train"),  # 训练集路径
#     eval_path=os.path.join(DATASET_BASE_DIR, "dpo_chemical", "eval"),    # 验证集路径
#     repo_id="/path/to/your/dpo/dataset"  # 数据集的repo_id
# )

# DPO数据集注册
_register_dataset(
    name="dpo_chemical",
    type=DatasetType.CHAT,
    train_path="/home/liyanhao/chemllm/REACT/logs/full/forward_reaction_prediction/epoch-/dpo_dataset.json"
)

# 如果需要混合数据集
_register_mixture(
    mixture_name="dpo_mixture",
    dataset_names={
        "dpo_chemical": 1.0,
    }
)

## Register a mixture of datasets
_register_mixture(
    mixture_name = "pubchem_cap",
    dataset_names = {"pubchem_cap":1.0},
)

_register_mixture(
    mixture_name = "uspto_rxn_interleaved",
    dataset_names = {"uspto_rxn":1.0},
)

_register_mixture(
    mixture_name = "pretrain_v2",
    dataset_names = {"uspto_rxn":1.0, "g2s":1.0, "s2f":1.0, "s2i":1.0},
)

_register_mixture(
    mixture_name = "pretrain_v3",
    dataset_names = {"uspto_rxn":1.0, "g2s":1.0, "s2f":1.0, "s2i":1.0, "i2s":1.0, "i2f":1.0},
)

_register_mixture(
    mixture_name = "pretrain_v3_01",
    dataset_names = {"uspto_rxn":0.05, "g2s":0.05, "s2f":0.05, "s2i":0.05, "i2s":0.05, "i2f":0.05, "trans":1.0},
)

_register_mixture(
    mixture_name = "pretrain_trans",
    dataset_names = {"trans":1.0},
)

_register_mixture(
    mixture_name = "sft",
    dataset_names = {
        "yields_regression": 1.0,
        "forward_prediction": 1.0,
        "retrosynthesis": 1.0,
        "reaction_classification": 1.0,
        "reagent_selection": 1.0,
        "reagent_prediction": 1.0,
        "solvent_prediction": 1.0,
        "catalyst_prediction": 1.0,
    },
)

_register_mixture(
    mixture_name = "sft_subset",
    dataset_names = {
        "yields_regression": 1.0, # ~9.5k
        "forward_prediction": 0.1, # ~12k
        "retrosynthesis": 0.1, # ~12k
        "reaction_classification": 0.1, # ~54k
        "reagent_selection": 1.0, # ~4k
        "reagent_prediction": 0.2, # ~11k
        "solvent_prediction": 0.2, # ~14k
        "catalyst_prediction": 1.0, # ~10k
    },
)

_register_mixture(
    mixture_name = "sft_subset_01",
    dataset_names = {
        "yields_regression": 0.02, # ~9.5k
        "forward_prediction": 0.02, # ~12k
        "retrosynthesis": 0.02, # ~12k
        "reaction_classification": 0.02, # ~54k
        "reagent_selection": 0.02, # ~4k
        "reagent_prediction": 0.02, # ~11k
        "solvent_prediction": 0.02, # ~14k
        "catalyst_prediction": 0.05, # ~10k
        "trans": 1.0
    },
)

_register_mixture(
    mixture_name = "nc",
    dataset_names = {
        "s2f": 1.0,
        "s2i": 1.0,
        "i2s": 1.0,
        "i2f": 1.0,
    },
)

_register_mixture(
    mixture_name = "text_only",
    dataset_names = {"i2s": 1.0, "i2f": 1.0},
)

# Register DPO datasets
_register_dataset("forward_reaction_dpo", DatasetType.CHAT, 
                 train_path="/home/liyanhao/chemllm/REACT/logs/full/forward_reaction_prediction/epoch-/dpo_dataset.json",
                 eval_path="/home/liyanhao/chemllm/REACT/logs/full/forward_reaction_prediction/epoch-/dpo_dataset.json")

# Register the DPO mixture
_register_mixture("dpo_mixture", {
    "forward_reaction_dpo": 1.0,
})

# if __name__ == "__main__":
#     # from presto.data import make_supervised_data_module
#     # from presto.trainer_utils import safe_save_model_for_hf_trainer, save_model_metadata, load_model_and_tokenizer_for_training
    
    
    
#     # model, tokenizer = load_model_and_tokenizer_for_training(model_cls, model_args, training_args, modalities)
#     dataset = _DATASETS['pubchem_cap']
#     train_dataset = _resolve_dataset(dataset, split="train")
    
#     print(_MIXTURES)