from typing import List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from presto.language_models.base_model import (
    LMMMetaModel,
    LMMMetaForCausalLM,
)
from presto.constants import IGNORE_INDEX

try:
    import deepspeed
    from deepspeed import zero
    from deepspeed.runtime.zero import GatheredParameters
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

class LlamaLMMConfig(LlamaConfig):
    model_type = "llama-lmm"
    
class LlamaLMMModel(LMMMetaModel, LlamaModel):
    config_class = LlamaLMMConfig

    def __init__(self, config: LlamaLMMConfig):
        # 初始化父类
        super().__init__(config)

        # 验证 vocab_size 和 hidden_size
        if config.vocab_size <= 0 or config.hidden_size <= 0:
            raise ValueError(f"Invalid vocab_size ({config.vocab_size}) or hidden_size ({config.hidden_size}) in config!")

        # 检查是否在 DeepSpeed ZeRO-3 环境下
        is_zero3 = DEEPSPEED_AVAILABLE and hasattr(zero, 'is_initialized') and zero.is_initialized()
        
        # 初始化 embed_tokens，确保即使在DeepSpeed下也能正确初始化
        if not hasattr(self, "embed_tokens") or self.embed_tokens is None:
            logging.info(f"Creating new embedding layer with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
            self.embed_tokens = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id,
            )
            
            # 不在这里初始化权重，而是在LlamaLMMForCausalLM中进行
            if is_zero3:
                logging.info("DeepSpeed Zero-3 is initialized. Weight initialization will be handled by the parent class.")
            else:
                # 仅在非DeepSpeed或未初始化时执行
                self.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)
                logging.info(f"Embedding layer initialized locally with shape: {self.embed_tokens.weight.shape}")
        else:
            logging.warning("Embedding layer already initialized.")

        # 仅在非 ZeRO-3 环境下验证嵌入层权重形状
        if not is_zero3:
            if self.embed_tokens.weight.shape[0] <= 0 or self.embed_tokens.weight.shape[1] <= 0:
                raise ValueError(f"Embedding layer weight is not properly initialized! Current shape: {self.embed_tokens.weight.shape}")
        else:
            # 在 ZeRO-3 环境下，只记录日志而不抛出错误
            logging.info(f"In ZeRO-3 mode, embedding layer shape is: {self.embed_tokens.weight.shape} (may be empty due to parameter partitioning)")
    
class LlamaLMMForCausalLM(LlamaForCausalLM, LMMMetaForCausalLM):
    config_class = LlamaLMMConfig

    def __init__(self, config):
        # 初始化父类
        LlamaForCausalLM.__init__(self, config)

        # 确保在DeepSpeed ZeRO-3下嵌入层正确初始化
        is_zero3 = DEEPSPEED_AVAILABLE and hasattr(zero, 'is_initialized') and zero.is_initialized()
        
        # 创建模型实例
        self.model = LlamaLMMModel(config)
        
        if is_zero3:
            # 使用GatheredParameters确保权重在所有GPU上可见
            try:
                # 检查是否初始化了分布式环境
                if torch.distributed.is_initialized():
                    with zero.GatheredParameters(self.model.embed_tokens.weight, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            # 确保在rank 0上初始化权重
                            self.model.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)
                            logging.info(f"Successfully initialized embedding weights on rank 0: {self.model.embed_tokens.weight.shape}")
                else:
                    logging.warning("Torch distributed not initialized but using ZeRO-3. Initializing embeddings directly.")
                    self.model.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)
            except Exception as e:
                logging.error(f"Error initializing embedding layer: {str(e)}")
                # 不抛出异常，而是尝试继续
                logging.warning("Continuing despite embedding initialization error...")
        else:
            # 非 ZeRO-3 环境下，确保嵌入层已正确初始化
            if hasattr(self.model.embed_tokens.weight, 'data'):
                if self.model.embed_tokens.weight.shape[0] <= 0 or self.model.embed_tokens.weight.shape[1] <= 0:
                    self.model.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)
                    logging.info(f"Re-initialized embedding weights with shape: {self.model.embed_tokens.weight.shape}")

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化 lm_head 权重
        self.lm_head.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # 初始化其他属性
        self._initialization_done = False
        self._forward_setup_done = False
        self.vocab_size = config.vocab_size
        self.modalities = None

    def get_model(self) -> "LlamaLMMForCausalLM":
        return self.model
    
    def verify_model_if_needed(self):
        if not self._initialization_done:
            try:
                weight_shape = self.model.embed_tokens.weight.shape
                logging.info(f"Model vocabulary size: {self.vocab_size}")
                logging.info(f"Embedding layer exists with shape: {weight_shape}")

                # 检查权重形状
                if weight_shape[0] != self.vocab_size or weight_shape[1] != self.config.hidden_size:
                    raise ValueError(
                        f"Embedding layer weight shape mismatch! Expected: ({self.vocab_size}, {self.config.hidden_size}), "
                        f"Found: {weight_shape}"
                    )
                    
                if weight_shape[0] == 0 or weight_shape[1] == 0:
                    raise ValueError(f"Embedding layer weight is not properly initialized! Current shape: {weight_shape}")

                self._initialization_done = True
            except Exception as e:
                logging.error(f"Model verification failed: {str(e)}")
                raise
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_modality_processing: bool = True,  # 添加跳过模态处理的标志
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.verify_model_if_needed()

        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if labels is not None:
            labels = labels.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)

        training_mode = kwargs.get('training_mode')
        is_dpo = training_mode == 'dpo'

        if is_dpo:
            try:
                # 在DPO模式下，我们需要分别处理chosen和rejected序列
                # 安全地计算batch_size
                if input_ids is not None and input_ids.shape[0] > 0:
                    batch_size = input_ids.shape[0] // 2
                elif inputs_embeds is not None and inputs_embeds.shape[0] > 0:
                    batch_size = inputs_embeds.shape[0] // 2
                else:
                    logging.error("DPO forward pass: Both input_ids and inputs_embeds cannot be None or empty")
                    # 直接调用父类的forward方法，跳过DPO处理
                    return super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict
                    )
                
                logging.info(f"DPO forward pass with batch size: {batch_size}")
                
                # 分割输入数据为chosen和rejected部分
                if input_ids is not None:
                    chosen_input_ids, rejected_input_ids = input_ids.chunk(2)
                    chosen_attention_mask, rejected_attention_mask = attention_mask.chunk(2) if attention_mask is not None else (None, None)
                    chosen_position_ids, rejected_position_ids = position_ids.chunk(2) if position_ids is not None else (None, None)
                    chosen_labels, rejected_labels = labels.chunk(2) if labels is not None else (None, None)
                    chosen_inputs_embeds, rejected_inputs_embeds = None, None
                else:
                    chosen_input_ids, rejected_input_ids = None, None
                    chosen_inputs_embeds, rejected_inputs_embeds = inputs_embeds.chunk(2)
                
                # 初始化这些变量，确保它们不会未定义
                chosen_projected_tensors = None
                rejected_projected_tensors = None
                chosen_past_key_values = self._clone_past_key_values(past_key_values) if past_key_values is not None else None
                rejected_past_key_values = self._clone_past_key_values(past_key_values) if past_key_values is not None else None
                
                # DPO模式下，只有在不跳过模态处理时才处理模态数据
                if not skip_modality_processing:
                    # 为chosen和rejected分别准备多模态输入
                    chosen_kwargs = {k: v for k, v in kwargs.items()}
                    rejected_kwargs = {k: v for k, v in kwargs.items()}
                    
                    # 如果有multimodal_inputs，也需要分割
                    for key in list(kwargs.keys()):
                        if isinstance(kwargs[key], torch.Tensor):
                            input_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
                            if kwargs[key].size(0) == input_size:
                                try:
                                    chosen_kwargs[key], rejected_kwargs[key] = kwargs[key].chunk(2)
                                    logging.info(f"Successfully split {key} tensor for DPO with shape: {kwargs[key].shape}")
                                except Exception as e:
                                    logging.warning(f"Failed to split {key} tensor for DPO: {str(e)}")
                                    # 如果分割失败，为chosen和rejected提供相同的值
                                    chosen_kwargs[key] = kwargs[key]
                                    rejected_kwargs[key] = kwargs[key]
                        elif isinstance(kwargs[key], dict):
                            # 处理嵌套字典
                            chosen_kwargs[key] = {}
                            rejected_kwargs[key] = {}
                            for sub_key, sub_value in kwargs[key].items():
                                if isinstance(sub_value, torch.Tensor):
                                    input_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
                                    if sub_value.size(0) == input_size:
                                        try:
                                            chosen_sub, rejected_sub = sub_value.chunk(2)
                                            chosen_kwargs[key][sub_key] = chosen_sub
                                            rejected_kwargs[key][sub_key] = rejected_sub
                                        except Exception as e:
                                            logging.warning(f"Failed to split nested {key}.{sub_key} tensor for DPO: {str(e)}")
                                            chosen_kwargs[key][sub_key] = sub_value
                                            rejected_kwargs[key][sub_key] = sub_value
                                    else:
                                        chosen_kwargs[key][sub_key] = sub_value
                                        rejected_kwargs[key][sub_key] = sub_value
                                else:
                                    chosen_kwargs[key][sub_key] = sub_value
                                    rejected_kwargs[key][sub_key] = sub_value
                    
                    # 分别处理chosen和rejected的多模态输入
                    try:
                        chosen_prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                            chosen_input_ids, chosen_attention_mask, 
                            chosen_past_key_values, 
                            chosen_labels, **chosen_kwargs
                        )
                        
                        if len(chosen_prepared_inputs) == 5:
                            chosen_input_ids, chosen_attention_mask, chosen_past_key_values, chosen_inputs_embeds, chosen_labels = chosen_prepared_inputs
                            chosen_projected_tensors = None
                        else:
                            chosen_input_ids, chosen_attention_mask, chosen_past_key_values, chosen_inputs_embeds, chosen_labels, chosen_projected_tensors = chosen_prepared_inputs
                        
                        rejected_prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                            rejected_input_ids, rejected_attention_mask, 
                            rejected_past_key_values, 
                            rejected_labels, **rejected_kwargs
                        )
                        
                        if len(rejected_prepared_inputs) == 5:
                            rejected_input_ids, rejected_attention_mask, rejected_past_key_values, rejected_inputs_embeds, rejected_labels = rejected_prepared_inputs
                            rejected_projected_tensors = None
                        else:
                            rejected_input_ids, rejected_attention_mask, rejected_past_key_values, rejected_inputs_embeds, rejected_labels, rejected_projected_tensors = rejected_prepared_inputs
                        
                    except Exception as e:
                        logging.error(f"Error in prepare_inputs_labels_for_multimodal for DPO: {str(e)}")
                        # 如果发生错误，重置变量并使用原始输入
                        chosen_projected_tensors, rejected_projected_tensors = None, None
                        chosen_inputs_embeds, rejected_inputs_embeds = None, None
                
                # 确保我们有有效的输入，即使是空的
                if (chosen_input_ids is None and chosen_inputs_embeds is None) or (rejected_input_ids is None and rejected_inputs_embeds is None):
                    # 创建一个最小的有效输入
                    chosen_input_ids = torch.tensor([[self.config.pad_token_id]], device=self.device)
                    rejected_input_ids = torch.tensor([[self.config.pad_token_id]], device=self.device)
                    chosen_attention_mask = torch.tensor([[1]], device=self.device)
                    rejected_attention_mask = torch.tensor([[1]], device=self.device)
                    if chosen_labels is not None:
                        chosen_labels = torch.tensor([[IGNORE_INDEX]], device=self.device)
                        rejected_labels = torch.tensor([[IGNORE_INDEX]], device=self.device)
                
                # 处理chosen序列
                try:
                    with torch.cuda.amp.autocast(enabled=True):
                        chosen_outputs = self.model(
                            input_ids=chosen_input_ids,
                            attention_mask=chosen_attention_mask,
                            position_ids=chosen_position_ids if chosen_position_ids is not None else None,
                            past_key_values=chosen_past_key_values,
                            inputs_embeds=chosen_inputs_embeds,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=True,
                        )

                        chosen_hidden_states = chosen_outputs[0]
                        chosen_logits = self.lm_head(chosen_hidden_states)
                        chosen_logits = chosen_logits.float()
                except Exception as e:
                    logging.error(f"Error in chosen forward pass: {str(e)}")
                    # 在错误情况下，创建默认输出
                    batch_size = 1 if chosen_input_ids is None else chosen_input_ids.shape[0]
                    seq_len = 1
                    if chosen_input_ids is not None:
                        seq_len = chosen_input_ids.shape[1]
                    elif chosen_inputs_embeds is not None:
                        seq_len = chosen_inputs_embeds.shape[1]
                    
                    chosen_hidden_states = torch.zeros((batch_size, seq_len, self.config.hidden_size), 
                                                      device=self.device, dtype=torch.float32)
                    chosen_logits = torch.zeros((batch_size, seq_len, self.config.vocab_size), 
                                                device=self.device, dtype=torch.float32)
                    chosen_outputs = {
                        "last_hidden_state": chosen_hidden_states,
                        "past_key_values": None
                    }

                # 处理rejected序列
                try:
                    with torch.cuda.amp.autocast(enabled=True):
                        rejected_outputs = self.model(
                            input_ids=rejected_input_ids,
                            attention_mask=rejected_attention_mask,
                            position_ids=rejected_position_ids if rejected_position_ids is not None else None,
                            past_key_values=rejected_past_key_values,
                            inputs_embeds=rejected_inputs_embeds,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=True,
                        )

                        rejected_hidden_states = rejected_outputs[0]
                        rejected_logits = self.lm_head(rejected_hidden_states)
                        rejected_logits = rejected_logits.float()
                except Exception as e:
                    logging.error(f"Error in rejected forward pass: {str(e)}")
                    # 在错误情况下，创建默认输出
                    batch_size = 1 if rejected_input_ids is None else rejected_input_ids.shape[0]
                    seq_len = 1
                    if rejected_input_ids is not None:
                        seq_len = rejected_input_ids.shape[1]
                    elif rejected_inputs_embeds is not None:
                        seq_len = rejected_inputs_embeds.shape[1]
                    
                    rejected_hidden_states = torch.zeros((batch_size, seq_len, self.config.hidden_size), 
                                                        device=self.device, dtype=torch.float32)
                    rejected_logits = torch.zeros((batch_size, seq_len, self.config.vocab_size), 
                                                  device=self.device, dtype=torch.float32)
                    rejected_outputs = {
                        "last_hidden_state": rejected_hidden_states,
                        "past_key_values": None
                    }
                
                # 计算chosen和rejected的损失
                chosen_loss = None
                if chosen_labels is not None:
                    shift_chosen_logits = chosen_logits[..., :-1, :].contiguous()
                    shift_chosen_labels = chosen_labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_chosen_logits = shift_chosen_logits.view(-1, self.config.vocab_size)
                    shift_chosen_labels = shift_chosen_labels.view(-1)
                    shift_chosen_labels = shift_chosen_labels.to(shift_chosen_logits.device)
                    chosen_loss = loss_fct(shift_chosen_logits, shift_chosen_labels)
                    if not skip_modality_processing and chosen_projected_tensors is not None:
                        chosen_loss = chosen_loss + chosen_projected_tensors.mean() * 0
                    chosen_loss = chosen_loss.mean()
                
                rejected_loss = None
                if rejected_labels is not None:
                    shift_rejected_logits = rejected_logits[..., :-1, :].contiguous()
                    shift_rejected_labels = rejected_labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_rejected_logits = shift_rejected_logits.view(-1, self.config.vocab_size)
                    shift_rejected_labels = shift_rejected_labels.view(-1)
                    shift_rejected_labels = shift_rejected_labels.to(shift_rejected_logits.device)
                    rejected_loss = loss_fct(shift_rejected_logits, shift_rejected_labels)
                    if not skip_modality_processing and rejected_projected_tensors is not None:
                        rejected_loss = rejected_loss + rejected_projected_tensors.mean() * 0
                    rejected_loss = rejected_loss.mean()
                
                # 返回DPO所需的输出
                return {
                    "chosen_logits": chosen_logits,
                    "chosen_loss": chosen_loss,
                    "rejected_logits": rejected_logits,
                    "rejected_loss": rejected_loss,
                    # 添加DPO特定的损失计算
                    "dpo_loss": self.compute_dpo_loss(
                        chosen_logits=chosen_logits,
                        chosen_labels=chosen_labels,
                        rejected_logits=rejected_logits,
                        rejected_labels=rejected_labels,
                        beta=kwargs.get("dpo_beta", 0.1),
                        ref_model=kwargs.get("ref_model"),
                        kwargs={
                            "ref_chosen_logits": kwargs.get("ref_chosen_logits"),
                            "ref_rejected_logits": kwargs.get("ref_rejected_logits"),
                            # 保持原有的inputs字段
                            "inputs": kwargs.get("inputs")
                        }
                    ) if chosen_labels is not None and rejected_labels is not None else None,
                    # 返回原始输出，以便于调试
                    "chosen_outputs": chosen_outputs,
                    "rejected_outputs": rejected_outputs
                }
            except Exception as e:
                logging.error(f"Error in DPO forward pass: {str(e)}")
                raise

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 非DPO模式下，只有在不跳过模态处理时才处理模态数据
        if not skip_modality_processing:
            try:
                prepared_inputs = self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, **kwargs
                )
                
                if len(prepared_inputs) == 5:
                    input_ids, attention_mask, past_key_values, inputs_embeds, labels = prepared_inputs
                    projected_tensors = None
                else:
                    input_ids, attention_mask, past_key_values, inputs_embeds, labels, projected_tensors = prepared_inputs
            except Exception as e:
                logging.error(f"Error in prepare_inputs_labels_for_multimodal: {str(e)}")
                projected_tensors = None
                inputs_embeds = None
        else:
            projected_tensors = None

        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        loss = None
        if labels is not None and training_mode != "dpo":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if not skip_modality_processing and projected_tensors is not None:
                loss = loss + projected_tensors.mean() * 0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        modality_inputs=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None:
            raise ValueError("inputs_embeds not supported")

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

        if modality_inputs:
            model_inputs.update(modality_inputs)

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is not None:
            reordered_past = ()
            for layer_past in past_key_values:
                reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) 
                                     for past_state in layer_past),)
            return reordered_past
        return None

    def _clone_past_key_values(self, past_key_values):
        """安全克隆 past_key_values 结构"""
        if past_key_values is None:
            return None
            
        try:
            # past_key_values 是一个元组的列表，每个元组包含两个张量
            cloned = []
            for layer_past in past_key_values:
                if isinstance(layer_past, tuple):
                    cloned.append(tuple(state.clone() for state in layer_past))
                else:
                    # 对于 past_key_values 可能有其他结构的情况提供支持
                    logging.warning(f"Found unexpected past_key_values structure: {type(layer_past)}")
                    cloned.append(layer_past)
            return cloned
        except Exception as e:
            logging.error(f"Error cloning past_key_values: {str(e)}")
            return past_key_values  # 如果克隆失败，返回原始值

    def compute_dpo_loss(self, chosen_logits, chosen_labels, rejected_logits, rejected_labels, beta=0.1, ref_model=None, kwargs=None):
        """
        计算DPO（Direct Preference Optimization）损失
        
        参数:
            chosen_logits: 首选回复的logits
            chosen_labels: 首选回复的标签
            rejected_logits: 被拒绝回复的logits
            rejected_labels: 被拒绝回复的标签
            beta: DPO温度参数，通常为0.1-0.5之间的值
            ref_model: 参考模型，用于计算KL散度正则化项
            kwargs: 其他参数，包含inputs字段用于传递给参考模型
        
        返回:
            DPO损失值
        """
        try:
            # 计算chosen和rejected的log概率
            chosen_log_probs = self._compute_sequence_log_probs(chosen_logits, chosen_labels)
            rejected_log_probs = self._compute_sequence_log_probs(rejected_logits, rejected_labels)
            
            # 检查是否已提供预计算的参考模型logits
            ref_chosen_logits = kwargs.get("ref_chosen_logits") if kwargs else None
            ref_rejected_logits = kwargs.get("ref_rejected_logits") if kwargs else None
            
            # 如果提供了预计算的logits，直接使用它们
            if ref_chosen_logits is not None and ref_rejected_logits is not None:
                ref_chosen_log_probs = self._compute_sequence_log_probs(ref_chosen_logits, chosen_labels)
                ref_rejected_log_probs = self._compute_sequence_log_probs(ref_rejected_logits, rejected_labels)
                
                # 计算KL散度项
                chosen_kl = (chosen_log_probs - ref_chosen_log_probs).mean()
                rejected_kl = (rejected_log_probs - ref_rejected_log_probs).mean()
                
                # 计算奖励差异
                reward_diff = (chosen_log_probs - ref_chosen_log_probs) - (rejected_log_probs - ref_rejected_log_probs)
            
            # 如果没有预计算的logits但提供了参考模型，则计算参考模型的输出
            elif ref_model is not None and hasattr(ref_model, 'forward') and kwargs and 'inputs' in kwargs:
                with torch.no_grad():
                    try:
                        # 使用相同的输入计算参考模型的输出
                        ref_inputs = kwargs.get('inputs', {})
                        
                        # 确保所有需要的键都存在
                        required_keys = [
                            'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
                            'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
                        ]
                        missing_keys = [k for k in required_keys if k not in ref_inputs]
                        
                        if missing_keys:
                            logging.warning(f"参考模型输入缺少必要的键: {missing_keys}")
                            # 如果缺少必要的键，则只使用政策模型的log概率
                            reward_diff = chosen_log_probs - rejected_log_probs
                        else:
                            # 添加训练模式
                            ref_inputs['training_mode'] = 'dpo'
                            
                            # 添加模型前向传播所需的所有键
                            ref_outputs = ref_model(
                                chosen_input_ids=ref_inputs['chosen_input_ids'],
                                chosen_attention_mask=ref_inputs['chosen_attention_mask'],
                                chosen_labels=ref_inputs['chosen_labels'],
                                rejected_input_ids=ref_inputs['rejected_input_ids'],
                                rejected_attention_mask=ref_inputs['rejected_attention_mask'],
                                rejected_labels=ref_inputs['rejected_labels'],
                                training_mode='dpo'
                            )
                            
                            # 提取logits
                            ref_chosen_logits = ref_outputs.get("chosen_logits")
                            ref_rejected_logits = ref_outputs.get("rejected_logits")
                            
                            if ref_chosen_logits is not None and ref_rejected_logits is not None:
                                ref_chosen_log_probs = self._compute_sequence_log_probs(ref_chosen_logits, chosen_labels)
                                ref_rejected_log_probs = self._compute_sequence_log_probs(ref_rejected_logits, rejected_labels)
                                
                                # 计算KL散度项
                                chosen_kl = (chosen_log_probs - ref_chosen_log_probs).mean()
                                rejected_kl = (rejected_log_probs - ref_rejected_log_probs).mean()
                                
                                # 计算奖励差异
                                reward_diff = (chosen_log_probs - ref_chosen_log_probs) - (rejected_log_probs - ref_rejected_log_probs)
                            else:
                                # 如果参考模型没有返回有效的logits，则直接使用政策模型的logits
                                logging.warning("参考模型未返回有效的logits，使用政策模型的logits")
                                reward_diff = chosen_log_probs - rejected_log_probs
                    except Exception as e:
                        logging.error(f"处理参考模型时出错: {str(e)}")
                        # 回退到仅使用政策模型
                        reward_diff = chosen_log_probs - rejected_log_probs
            else:
                # 计算chosen和rejected的奖励差异
                reward_diff = chosen_log_probs - rejected_log_probs
            
            # 计算DPO损失 (Sigmoid形式)
            loss = -torch.nn.functional.logsigmoid(beta * reward_diff).mean()
            
            # 记录输出
            with torch.no_grad():
                chosen_rewards = chosen_log_probs.mean().item()
                rejected_rewards = rejected_log_probs.mean().item()
                avg_reward_diff = reward_diff.mean().item()
                logging.info(f"DPO loss: {loss.item():.4f}, chosen: {chosen_rewards:.4f}, "
                             f"rejected: {rejected_rewards:.4f}, diff: {avg_reward_diff:.4f}")
            
            return loss
        except Exception as e:
            logging.error(f"计算DPO损失时出错: {str(e)}")
            # 返回一个默认损失值
            return torch.tensor(1.0, device=chosen_logits.device, requires_grad=True)

    def _compute_sequence_log_probs(self, logits, labels):
        """
        计算序列的对数概率
        
        参数:
            logits: 模型输出的logits [batch, seq_len, vocab]
            labels: 目标标签 [batch, seq_len]
        
        返回:
            每个序列的对数概率 [batch]
        """
        # 将logits和labels错开一位，这样我们预测的是下一个token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        
        # 获取每个位置的对数概率
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # 获取目标词的对数概率
        token_log_probs = torch.gather(
            log_probs.view(-1, log_probs.size(-1)), 
            1, 
            shift_labels.view(-1, 1)
        ).view(shift_labels.size())
        
        # 创建掩码，忽略padding位置
        mask = (shift_labels != self.config.pad_token_id).float()
        
        # 计算有效位置的对数概率和
        seq_log_probs = (token_log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return seq_log_probs

    def prepare_dpo_inputs(self, batch):
        """
        处理DPO批次数据，将其转换为模型所需的格式
        
        参数:
            batch: 包含chosen和rejected数据的批次
        
        返回:
            处理后的模型输入
        """
        try:
            # 从批次中提取数据
            molecules = batch.get("molecules", {})
            smiles = molecules.get("smiles", [])
            molecule_count = len(smiles) if isinstance(smiles, list) else 0
            
            # 处理chosen和rejected消息
            chosen_messages = batch.get("chosen_messages", [])
            rejected_messages = batch.get("rejected_messages", [])
            
            # 将消息转换为模型输入格式
            chosen_input_texts = []
            rejected_input_texts = []
            
            # 处理对话历史
            for chosen, rejected in zip(chosen_messages, rejected_messages):
                # 替换<molecule_2d>标记为实际分子表示
                chosen_text = chosen.get("content", "")
                rejected_text = rejected.get("content", "")
                
                for i in range(molecule_count):
                    mol_placeholder = f"<molecule_2d>"
                    if i < molecule_count and mol_placeholder in chosen_text:
                        chosen_text = chosen_text.replace(mol_placeholder, smiles[i], 1)
                    
                    if i < molecule_count and mol_placeholder in rejected_text:
                        rejected_text = rejected_text.replace(mol_placeholder, smiles[i], 1)
                
                chosen_input_texts.append(chosen_text)
                rejected_input_texts.append(rejected_text)
            
            # 对文本进行分词，获取模型输入
            chosen_inputs = self.tokenizer(chosen_input_texts, padding=True, truncation=True, return_tensors="pt")
            rejected_inputs = self.tokenizer(rejected_input_texts, padding=True, truncation=True, return_tensors="pt")
            
            # 合并inputs，将chosen和rejected输入连接起来
            combined_inputs = {
                "input_ids": torch.cat([chosen_inputs["input_ids"], rejected_inputs["input_ids"]], dim=0),
                "attention_mask": torch.cat([chosen_inputs["attention_mask"], rejected_inputs["attention_mask"]], dim=0),
            }
            
            # 添加多模态特定输入
            modality_inputs = {}
            for i, smile in enumerate(smiles):
                modality_inputs[f"molecule_{i}"] = smile
            
            combined_inputs["modality_inputs"] = modality_inputs
            combined_inputs["labels"] = combined_inputs["input_ids"].clone()
            
            return combined_inputs
        except Exception as e:
            logging.error(f"Error in prepare_dpo_inputs: {str(e)}")
            raise

# Register the model
AutoConfig.register("llama-lmm", LlamaLMMConfig)
AutoModelForCausalLM.register(LlamaLMMConfig, LlamaLMMForCausalLM)