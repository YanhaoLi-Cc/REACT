import os
import json
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import Trainer, TrainingArguments

from .modalities import Modality
from .language_models.llama import LlamaLMMForCausalLM


class DPODataset(Dataset):
    """
    用于 DPO 训练的数据集类
    
    处理 DPO 特定的数据格式，将其转换为模型输入格式
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        modalities: List[Modality],
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        """
        初始化 DPO 数据集
        
        参数:
            data_path: DPO 数据文件路径
            tokenizer: 用于分词的分词器
            modalities: 模型支持的模态列表
            max_length: 输入序列的最大长度
            max_prompt_length: 提示部分的最大长度
        """
        self.tokenizer = tokenizer
        self.modalities = modalities
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        self.raw_data = self._load_data(data_path)
        logging.info(f"Loaded {len(self.raw_data)} examples from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """
        加载 DPO 数据
        
        参数:
            data_path: 数据文件路径
            
        返回:
            加载的数据列表
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.raw_data)
    
    def _process_messages(self, messages: List[Dict], molecules_data: Dict) -> str:
        """
        处理对话消息列表，替换分子标记为实际的分子表示
        
        参数:
            messages: 消息列表
            molecules_data: 分子数据
            
        返回:
            处理后的对话文本
        """
        smiles = molecules_data.get("smiles", [])
        
        # 构建对话文本
        dialog_text = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # 替换分子标记为实际的分子表示
            for i in range(len(smiles)):
                mol_tag = f"<molecule_2d>"
                if mol_tag in content:
                    mol_repr = smiles[i] if i < len(smiles) else ""
                    content = content.replace(mol_tag, mol_repr, 1)
            
            if role == "system":
                dialog_text += f"System: {content}\n"
            elif role == "user":
                dialog_text += f"User: {content}\n"
            elif role == "assistant":
                dialog_text += f"Assistant: {content}\n"
                
        return dialog_text.strip()
        
    def __getitem__(self, idx: int) -> Dict:
        """
        获取指定索引的数据样本
        
        参数:
            idx: 数据索引
            
        返回:
            处理后的数据样本
        """
        example = self.raw_data[idx]
        
        # 提取数据
        molecules_data = example.get("molecules", {})
        chosen_messages = example.get("chosen_messages", [])
        rejected_messages = example.get("rejected_messages", [])
        
        # 处理对话消息
        chosen_dialog = self._process_messages(chosen_messages, molecules_data)
        rejected_dialog = self._process_messages(rejected_messages, molecules_data)
        
        # 对文本进行分词
        chosen_tokens = self.tokenizer(
            chosen_dialog,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        rejected_tokens = self.tokenizer(
            rejected_dialog,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 准备模型输入
        model_inputs = {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }
        
        # 添加分子数据
        for m in self.modalities:
            if m.data_key in molecules_data:
                model_inputs[m.name] = molecules_data[m.data_key]
        
        return model_inputs


class DPODataCollator:
    """
    用于 DPO 训练的数据收集器
    
    处理批次数据，确保模态数据正确合并
    """
    
    def __init__(self, tokenizer, modalities: List[Modality], pad_to_multiple_of=None):
        """
        初始化 DPO 数据收集器
        
        参数:
            tokenizer: 分词器
            modalities: 模态列表
            pad_to_multiple_of: 填充至倍数
        """
        self.tokenizer = tokenizer
        self.modalities = modalities
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        """
        将特征列表转换为批次
        
        参数:
            features: 特征列表
            
        返回:
            批次数据
        """
        # 检查特征列表是否为空
        if not features:
            return {}
            
        # 提取所有键
        keys = set(features[0].keys())
        
        batch = {}
        
        # 处理文本输入
        for key in ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]:
            if key in keys:
                batch[key] = torch.stack([f[key] for f in features])
                
        # 处理模态数据
        for m in self.modalities:
            if m.name in keys:
                # 根据模态类型处理数据
                if m.tensor_based:
                    # 对于基于张量的模态，使用 stack 或 cat 操作
                    batch[m.name] = m.collate_fn([f[m.name] for f in features])
                else:
                    # 对于非张量模态，直接收集
                    batch[m.name] = [f[m.name] for f in features]
        
        return batch


class DPOTrainer(Trainer):
    """
    用于 DPO (Direct Preference Optimization) 训练的训练器
    
    扩展 HuggingFace Trainer 以支持 DPO 训练逻辑
    """
    
    def __init__(
        self,
        model: LlamaLMMForCausalLM,
        ref_model: Optional[LlamaLMMForCausalLM] = None,
        args: TrainingArguments = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        modalities: List[Modality] = None,
        **kwargs,
    ):
        # 创建默认的数据收集器
        if data_collator is None and modalities is not None and tokenizer is not None:
            data_collator = DPODataCollator(tokenizer, modalities)
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )
        
        # 保存模态信息
        self.modalities = modalities
        
        # 参考模型，用于 DPO 训练
        self.ref_model = ref_model
        
        # DPO 特定参数
        self.beta = kwargs.pop("beta", 0.1)
        self.max_prompt_length = kwargs.pop("max_prompt_length", 512)
        self.max_length = kwargs.pop("max_length", 2048)
        
        # 确保模型和参考模型都进入正确的训练模式
        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
            
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算 DPO 损失
        
        参数:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
            
        返回:
            损失值，以及可选的输出
        """
        # 设置训练模式参数
        kwargs = {
            "training_mode": "dpo",
            "dpo_beta": self.beta,
        }
        
        # 为 chosen 和 rejected 序列准备输入
        chosen_input_ids = inputs.get("chosen_input_ids")
        chosen_attention_mask = inputs.get("chosen_attention_mask")
        rejected_input_ids = inputs.get("rejected_input_ids")  
        rejected_attention_mask = inputs.get("rejected_attention_mask")
        
        # 合并输入
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        labels = input_ids.clone()
        
        # 收集模态输入
        modality_inputs = {}
        for key, value in inputs.items():
            if key not in ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]:
                modality_inputs[key] = value
        
        # 模型前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **modality_inputs,
            **kwargs,
        )
        
        # 获取 DPO 损失
        loss = outputs.get("dpo_loss")
        
        if return_outputs:
            return loss, outputs
        return loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        """
        保存模型
        
        参数:
            output_dir: 输出目录，如果为 None 则使用 self.args.output_dir
            _internal_call: 是否为内部调用
        """
        if output_dir is None:
            output_dir = self.args.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        
        # 保存分词器
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        # 保存训练参数
        self.save_state()
        
    @classmethod
    def load_dataset(cls, data_path: str, tokenizer, modalities: List[Modality], max_length: int = 2048, max_prompt_length: int = 512) -> DPODataset:
        """
        加载 DPO 数据集
        
        参数:
            data_path: 数据文件路径
            tokenizer: 分词器
            modalities: 模态列表
            max_length: 最大序列长度
            max_prompt_length: 最大提示长度
            
        返回:
            DPO 数据集
        """
        return DPODataset(
            data_path=data_path,
            tokenizer=tokenizer,
            modalities=modalities,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        ) 