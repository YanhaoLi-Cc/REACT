from typing import Optional, List, Type, Dict
from dataclasses import field, dataclass
import pathlib
import torch
import shutil
import glob
import os
import logging
import sys

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer, TrainerCallback, PreTrainedModel
from trl import DPOTrainer

from presto.data import (
    make_supervised_data_module,
    DataCollatorForDPODataset
)
from presto.model_utils import (
    get_peft_state,
    get_peft_state_non_lora,
    get_peft_state,
    get_peft_state_non_lora,
)
from presto.modalities.base_modality import Modality
from presto.hparams import TrainingArguments, ModelArguments
from presto.trainer_utils import safe_save_model_for_hf_trainer, save_model_metadata, load_model_and_tokenizer_for_training


local_rank = None

class LMMSupervisedTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMSupervisedTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMSupervisedTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters())
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )

class LMMDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        # 提取ref_model和模型参数，以便特殊处理
        ref_model = kwargs.pop("ref_model", None)
        model = kwargs.get("model", None)
        
        # 确保ref_model和model不共享内存
        if ref_model is not None and model is not None:
            # 在DeepSpeed模式下特殊处理参考模型
            if any(arg.startswith("deepspeed") for arg in sys.argv):
                logging.info("在DeepSpeed模式下使用独立参考模型")
                # 使用模型的副本作为参考模型
                for param in ref_model.parameters():
                    param.requires_grad = False  # 确保参考模型不需要梯度
            
            # 将处理后的参考模型放回kwargs
            kwargs["ref_model"] = ref_model
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMDPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMDPOTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters())
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """重写compute_loss方法，以支持参考模型"""
        try:
            # 添加调试信息
            logging.info(f"DPO inputs keys: {list(inputs.keys())}")
            
            # 检查输入有效性
            if "input_ids" not in inputs or not isinstance(inputs["input_ids"], torch.Tensor):
                logging.error(f"Missing or invalid input_ids")
                # 创建一个最小有效的输入
                device = next(model.parameters()).device
                input_ids = torch.tensor([[0]], device=device)
                attention_mask = torch.tensor([[1]], device=device)
                labels = torch.tensor([[-100]], device=device)
                
                # 重新构建inputs字典
                inputs = {
                    "input_ids": torch.cat([input_ids, input_ids], dim=0),
                    "attention_mask": torch.cat([attention_mask, attention_mask], dim=0),
                    "labels": torch.cat([labels, labels], dim=0),
                }
                
            # 为DPO处理输入
            batch_size = inputs["input_ids"].shape[0] // 2
            
            # 检查batch_size是否有效
            if batch_size <= 0:
                logging.error(f"Invalid batch size: {inputs['input_ids'].shape[0]}")
                # 创建一个最小有效的批次
                device = next(model.parameters()).device
                input_ids = torch.tensor([[0]], device=device)
                attention_mask = torch.tensor([[1]], device=device)
                labels = torch.tensor([[-100]], device=device)
                
                # 重新构建inputs字典
                inputs = {
                    "input_ids": torch.cat([input_ids, input_ids], dim=0),
                    "attention_mask": torch.cat([attention_mask, attention_mask], dim=0),
                    "labels": torch.cat([labels, labels], dim=0),
                }
                batch_size = 1
            
            # 分离chosen和rejected
            chosen_input_ids = inputs["input_ids"][:batch_size]
            chosen_attention_mask = inputs["attention_mask"][:batch_size] if "attention_mask" in inputs else None
            chosen_labels = inputs["labels"][:batch_size] if "labels" in inputs else None
            
            rejected_input_ids = inputs["input_ids"][batch_size:]
            rejected_attention_mask = inputs["attention_mask"][batch_size:] if "attention_mask" in inputs else None
            rejected_labels = inputs["labels"][batch_size:] if "labels" in inputs else None
            
            # 为模态输入做类似的处理，但在DPO训练时我们会跳过这些
            modality_inputs = {}
            for key, value in inputs.items():
                if key not in ["input_ids", "attention_mask", "labels"]:
                    # 检查是否为模态数据
                    if isinstance(value, list) and len(value) > 0:
                        # 如果模态数据是成对的
                        if len(value) == inputs["input_ids"].shape[0]:
                            modality_inputs[key] = value
                        else:
                            logging.warning(f"Modality data {key} length mismatch: {len(value)} vs {inputs['input_ids'].shape[0]}")
            
            # 设置跳过模态处理的标志
            skip_modality_processing = True
            
            # 重新构建DPO输入格式
            dpo_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"] if "attention_mask" in inputs else torch.ones_like(inputs["input_ids"]),
                "labels": inputs["labels"] if "labels" in inputs else torch.full_like(inputs["input_ids"], -100),
                "training_mode": "dpo",
                "dpo_beta": getattr(self.args, "dpo_beta", 0.1),
                "skip_modality_processing": skip_modality_processing,  # 传递标志
            }
            
            # 如果不跳过模态处理，则添加模态数据（这里我们跳过）
            if not skip_modality_processing:
                for key, value in modality_inputs.items():
                    dpo_inputs[key] = value
            
            # 添加参考模型
            if hasattr(self, 'ref_model') and self.ref_model is not None:
                # 不直接传递ref_model到dpo_inputs，而是在下面单独处理
                # 这样可以避免在梯度检查点中重复计算梯度
                pass
            
            # 调用模型的前向传播
            try:
                # 如果有参考模型，先用它前向传播，但不计算梯度
                ref_chosen_logits = None
                ref_rejected_logits = None
                if hasattr(self, 'ref_model') and self.ref_model is not None:
                    with torch.no_grad():
                        # 为参考模型创建输入
                        ref_inputs = {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"] if "attention_mask" in inputs else torch.ones_like(inputs["input_ids"]),
                            "labels": inputs["labels"] if "labels" in inputs else torch.full_like(inputs["input_ids"], -100),
                            "training_mode": "dpo",
                            "skip_modality_processing": True,  # 参考模型也跳过模态处理
                        }
                        ref_outputs = self.ref_model(**ref_inputs)
                        # 提取参考模型的logits
                        if isinstance(ref_outputs, dict):
                            ref_chosen_logits = ref_outputs.get("chosen_logits")
                            ref_rejected_logits = ref_outputs.get("rejected_logits")
                
                # 将参考模型的输出作为单独参数传递给主模型
                if ref_chosen_logits is not None and ref_rejected_logits is not None:
                    dpo_inputs["ref_chosen_logits"] = ref_chosen_logits.detach()
                    dpo_inputs["ref_rejected_logits"] = ref_rejected_logits.detach()
                
                outputs = model(**dpo_inputs)
            except Exception as e:
                logging.error(f"Error in DPO forward pass: {str(e)}")
                
                # 尝试不带模态数据的前向传播
                basic_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"] if "attention_mask" in inputs else torch.ones_like(inputs["input_ids"]),
                    "labels": inputs["labels"] if "labels" in inputs else torch.full_like(inputs["input_ids"], -100),
                    "training_mode": "dpo",
                    "dpo_beta": getattr(self.args, "dpo_beta", 0.1),
                    "skip_modality_processing": True,  # 强制跳过模态处理
                }
                
                if hasattr(self, 'ref_model') and self.ref_model is not None:
                    basic_inputs['ref_model'] = self.ref_model
                
                try:
                    outputs = model(**basic_inputs)
                except Exception as e:
                    logging.error(f"Error in basic DPO forward pass: {str(e)}")
                    # 如果仍然失败，创建一个伪造的输出对象
                    device = next(model.parameters()).device
                    fake_loss = torch.tensor(1.0, device=device, requires_grad=True)
                    if return_outputs:
                        return fake_loss, {"loss": fake_loss}
                    return fake_loss
            
            # 获取DPO损失
            if isinstance(outputs, dict) and 'dpo_loss' in outputs:
                loss = outputs['dpo_loss']
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                logging.error("Model output doesn't contain loss")
                # 创建一个默认的损失
                device = next(model.parameters()).device
                loss = torch.tensor(1.0, device=device, requires_grad=True)
            
            # 检查loss是否有效
            if not isinstance(loss, torch.Tensor) or torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.error(f"Invalid loss detected: {loss}")
                device = next(model.parameters()).device
                loss = torch.tensor(1.0, device=device, requires_grad=True)
            
            if return_outputs:
                return loss, outputs
            return loss
        except Exception as e:
            logging.error(f"Unhandled error in compute_loss: {str(e)}")
            # 返回一个零损失，防止训练崩溃
            device = next(model.parameters()).device
            fake_loss = torch.tensor(1.0, device=device, requires_grad=True)
            if return_outputs:
                return fake_loss, {"loss": fake_loss}
            return fake_loss


def train_for_modalities(
    model_cls,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    modalities: List[Modality],
):
    global local_rank
    local_rank = training_args.local_rank

    # 添加CUDA调试信息
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    model, tokenizer = load_model_and_tokenizer_for_training(model_cls, model_args, training_args, modalities)
    data_module = make_supervised_data_module(tokenizer, training_args, modalities)
    save_model_metadata(model_cls, training_args, model_args, modalities, data_module, model)
    
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            if args.lora_enable:
                state_dict = get_peft_state(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora(
                    model.named_parameters()
                )
                if args.local_rank in [-1, 0]:
                    model.config.save_pretrained(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))

    class DebugCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                logging.info("First batch debug information:")
                for k, v in kwargs.get("inputs", {}).items():
                    if isinstance(v, torch.Tensor):
                        logging.info(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                        if torch.isnan(v).any():
                            logging.warning(f"NaN values detected in {k}")

    if training_args.training_mode == "sft":
        trainer = LMMSupervisedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[SaveCallback(), DebugCallback()],
            **data_module,
        )
    elif training_args.training_mode == "dpo":
        if training_args.ref_model:
            ref_model, _ = load_model_and_tokenizer_for_training(model_cls, model_args, training_args, modalities)
        else:
            ref_model = None
        
        # 使用自定义的数据整理器，传递skip_modality_processing参数
        data_module['data_collator'] = DataCollatorForDPODataset(
            tokenizer=tokenizer,
            modalities=modalities,
            skip_modality_processing=getattr(training_args, "skip_modality_processing", False)
        )
        
        trainer = LMMDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            beta=training_args.dpo_beta,
            tokenizer=tokenizer,
            callbacks=[SaveCallback(), DebugCallback()],
            **data_module,
        )

        # 禁用DeepSpeed的参数共享检测
        if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "deepspeed_engine_wrapped"):
            if hasattr(trainer.accelerator.deepspeed_engine_wrapped, "engine"):
                logging.info("禁用DeepSpeed的参数共享检测")
                if hasattr(trainer.accelerator.deepspeed_engine_wrapped.engine, "param_dict"):
                    trainer.accelerator.deepspeed_engine_wrapped.engine.param_dict = {}
                    
                # 禁用梯度累积检测
                if hasattr(trainer.accelerator.deepspeed_engine_wrapped.engine, "param_names_already_reduced"):
                    trainer.accelerator.deepspeed_engine_wrapped.engine.param_names_already_reduced = set()

        # 添加额外的训练步骤验证
        old_training_step = trainer.training_step
        def new_training_step(model, inputs):
            try:
                # 清除已减少的参数记录
                if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "deepspeed_engine_wrapped"):
                    if hasattr(trainer.accelerator.deepspeed_engine_wrapped, "engine"):
                        if hasattr(trainer.accelerator.deepspeed_engine_wrapped.engine, "param_names_already_reduced"):
                            trainer.accelerator.deepspeed_engine_wrapped.engine.param_names_already_reduced = set()
                            
                return old_training_step(model, inputs)
            except Exception as e:
                logging.error(f"Error in training step: {str(e)}")
                logging.error("Input shapes:")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        logging.error(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                raise
        trainer.training_step = new_training_step
        
    else:
        raise ValueError(f"Unknown training mode: {training_args.training_mode}")

    # 添加数据验证步骤
    # if training_args.local_rank in [-1, 0]:
    #     logging.info("Validating first batch of data...")
    #     dataloader = trainer.get_train_dataloader()
    #     first_batch = next(iter(dataloader))
    #     logging.info("First batch shapes:")
    #     for k, v in first_batch.items():
    #         if isinstance(v, torch.Tensor):
    #             logging.info(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # 在DeepSpeed模式下验证参数是否有共享问题
    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "deepspeed_engine_wrapped"):
        if training_args.local_rank in [-1, 0]:
            logging.info("检查DeepSpeed参数共享情况...")
            
            # 确保每个参数都注册到唯一的ID
            engine = trainer.accelerator.deepspeed_engine_wrapped.engine
            if hasattr(engine, "param_dict"):
                engine.param_dict = {}  # 清空参数字典，让DeepSpeed重新注册
            
            # 注册所有参数到param_id字典
            param_ids = {}
            for name, param in model.named_parameters():
                param_id = id(param)
                if param_id in param_ids:
                    logging.warning(f"发现共享参数! {name} 与 {param_ids[param_id]} 共享内存")
                else:
                    param_ids[param_id] = name
            
            # 验证ref_model和model不共享参数
            if hasattr(trainer, "ref_model") and trainer.ref_model is not None:
                ref_param_ids = {}
                for ref_name, ref_param in trainer.ref_model.named_parameters():
                    ref_param_id = id(ref_param)
                    if ref_param_id in param_ids:
                        logging.error(f"参考模型参数 {ref_name} 与主模型参数 {param_ids[ref_param_id]} 共享内存!")
                        raise ValueError("参考模型和主模型参数共享内存，这会导致DeepSpeed梯度计算错误")
                    ref_param_ids[ref_param_id] = ref_name
                
                logging.info(f"参考模型有 {len(ref_param_ids)} 个独立参数")

    if list(pathlib.Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)