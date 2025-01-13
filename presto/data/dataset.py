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



# @dataclass 
# class DataCollatorForDPODataset:
#     """Data collator for DPO training"""
#     tokenizer: transformers.PreTrainedTokenizer
#     modalities: List[Modality]
    
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         # 分别收集chosen和rejected序列
#         chosen_input_ids = [instance["chosen_input_ids"] for instance in instances]
#         chosen_labels = [instance["chosen_labels"] for instance in instances]
#         rejected_input_ids = [instance["rejected_input_ids"] for instance in instances]
#         rejected_labels = [instance["rejected_labels"] for instance in instances]

#         # Padding处理
#         chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
#             chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         chosen_labels = torch.nn.utils.rnn.pad_sequence(
#             chosen_labels, batch_first=True, padding_value=IGNORE_INDEX
#         )
#         rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
#             rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         rejected_labels = torch.nn.utils.rnn.pad_sequence(
#             rejected_labels, batch_first=True, padding_value=IGNORE_INDEX
#         )
        
#         # 截断处理
#         max_length = self.tokenizer.model_max_length
#         chosen_input_ids = chosen_input_ids[:, :max_length]
#         chosen_labels = chosen_labels[:, :max_length] 
#         rejected_input_ids = rejected_input_ids[:, :max_length]
#         rejected_labels = rejected_labels[:, :max_length]

#         batch = {
#             "chosen_input_ids": chosen_input_ids,
#             "chosen_labels": chosen_labels,
#             "chosen_attention_mask": chosen_input_ids.ne(self.tokenizer.pad_token_id),
#             "rejected_input_ids": rejected_input_ids,
#             "rejected_labels": rejected_labels,
#             "rejected_attention_mask": rejected_input_ids.ne(self.tokenizer.pad_token_id),
#         }

#         # 添加modality相关数据
#         for m in self.modalities:
#             batch[m.name] = [instance[m.name] for instance in instances]

#         return batch

@dataclass
class DPODataCollatorWithPadding:
    """
    Custom data collator for DPO training
    """
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # 处理文本数据
        for key in ["chosen_input_ids", "chosen_labels", "chosen_attention_mask",
                   "rejected_input_ids", "rejected_labels", "rejected_attention_mask"]:
            if key in features[0]:
                # 确保所有输入都是tensor
                sequence = [torch.tensor(f[key]) if not isinstance(f[key], torch.Tensor) else f[key] for f in features]
                
                # Padding处理
                sequence = pad_sequence(
                    sequence,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id if "input_ids" in key 
                    else (0 if "attention_mask" in key else -100)
                )
                
                # 截断处理
                if self.max_length:
                    sequence = sequence[:, :self.max_length]
                
                # pad to multiple of
                if self.pad_to_multiple_of:
                    seq_length = sequence.size(1)
                    padding_amount = (self.pad_to_multiple_of - seq_length % self.pad_to_multiple_of) % self.pad_to_multiple_of
                    if padding_amount > 0:
                        padding_value = (
                            self.tokenizer.pad_token_id if "input_ids" in key
                            else (0 if "attention_mask" in key else -100)
                        )
                        padding = torch.ones(sequence.size(0), padding_amount) * padding_value
                        sequence = torch.cat([sequence, padding.to(sequence.device)], dim=1)
                
                # 设置正确的数据类型
                if "input_ids" in key:
                    sequence = sequence.long()  # 确保input_ids是长整型
                elif "attention_mask" in key:
                    sequence = sequence.bool().float()  # attention_mask转换为布尔值再转浮点
                elif "labels" in key:
                    sequence = sequence.long()  # labels也应该是长整型
                
                batch[key] = sequence
        
        # 处理modality数据
        for m in self.modalities:
            if m.name in features[0]:
                modality_data = []
                max_len = max(len(f[m.name]) for f in features)
                
                for feature in features:
                    data = feature[m.name]
                    # Padding处理
                    if len(data) < max_len:
                        padded_data = data + [None] * (max_len - len(data))
                    else:
                        padded_data = data[:max_len]
                    modality_data.append(padded_data)
                
                batch[m.name] = modality_data
        
        return batch

    def _verify_labels(self, labels: torch.Tensor) -> None:
        """验证标签的有效性"""
        if labels.dim() != 2:
            raise ValueError(f"Labels should be 2D but got {labels.dim()}D tensor")
        if not ((labels >= -100).all() and (labels >= 0).any()):
            raise ValueError("Labels should be either -100 or positive integers")

@dataclass
class CustomDPODataCollator(DPODataCollatorWithPadding):
    """Extends DPODataCollatorWithPadding to handle modalities"""
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 首先使用父类的collate方法处理标准DPO数据
        batch = super().__call__(instances)
        
        # 处理modality数据
        for m in self.modalities:
            if m.name in instances[0]:
                modality_data = []
                max_len = max(len(instance[m.name]) for instance in instances)
                
                for instance in instances:
                    data = instance[m.name]
                    # Pad if necessary
                    if len(data) < max_len:
                        padded_data = data + [None] * (max_len - len(data))
                    else:
                        padded_data = data
                    modality_data.append(padded_data)
                
                batch[m.name] = modality_data
        
        return batch

@dataclass 
class DataCollatorForDPODataset:
    """Data collator for DPO training"""
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        collator = DPODataCollatorWithPadding(
            tokenizer=self.tokenizer,
            modalities=self.modalities,
            max_length=self.tokenizer.model_max_length,
            pad_to_multiple_of=8
        )
        return collator(instances)
    

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
        assert batch["chosen_attention_mask"].dtype == torch.bool, \
            "Attention mask should be boolean"
        
        # 检查input_ids的有效性
        assert (batch["chosen_input_ids"] >= 0).all(), "Invalid input ids found"
        assert (batch["rejected_input_ids"] >= 0).all(), "Invalid input ids found"

        # 检查labels的有效性
        valid_labels = (batch["chosen_labels"] == IGNORE_INDEX) | (batch["chosen_labels"] >= 0)
        assert valid_labels.all(), "Invalid labels found"

    
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
    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            encoded = encode_dpo_chat(item, self.tokenizer, self.modalities)
            
            # 构建DPO所需的格式
            result = {
                "chosen_input_ids": encoded["chosen_input_ids"],
                "chosen_labels": encoded["chosen_labels"],
                "chosen_attention_mask": encoded["chosen_input_ids"].ne(self.tokenizer.pad_token_id),
                "rejected_input_ids": encoded["rejected_input_ids"],
                "rejected_labels": encoded["rejected_labels"],
                "rejected_attention_mask": encoded["rejected_input_ids"].ne(self.tokenizer.pad_token_id),
            }
            
            # 添加modality数据
            for m in self.modalities:
                if m.name in encoded:
                    result[m.name] = encoded[m.name]
            
            return result
            
        except Exception as e:
            new_i = i + 1 if i + 1 < len(self) else 0
            logging.error(f"Error encoding DPO data: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)

    def _validate_encoded_data(self, encoded: Dict):
        """验证编码数据的有效性"""
        # 检查必需的键
        required_keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
        for key in required_keys:
            assert key in encoded, f"Missing required key: {key}"
            assert isinstance(encoded[key], torch.Tensor), f"{key} must be a tensor"
        
        # 检查modality数据
        for m in self.modalities:
            assert m.name in encoded, f"Missing modality data: {m.name}"
            assert isinstance(encoded[m.name], list), f"{m.name} must be a list"


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

# if __name__ == "__main__":
#     # from presto.data import make_supervised_data_module
#     # from presto.trainer_utils import safe_save_model_for_hf_trainer, save_model_metadata, load_model_and_tokenizer_for_training
    
    
    
#     # model, tokenizer = load_model_and_tokenizer_for_training(model_cls, model_args, training_args, modalities)
#     dataset = _DATASETS['pubchem_cap']
#     train_dataset = _resolve_dataset(dataset, split="train")
    
#     print(_MIXTURES)