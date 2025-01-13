
    
# class LlamaLMMConfig(LlamaConfig):
#     model_type = "llama-lmm"


# class LlamaLMMModel(LMMMetaModel, LlamaModel):
#     config_class = LlamaLMMConfig

#     def __init__(self, config: LlamaLMMConfig):
#         LlamaModel.__init__(self, config)  # 先调用LlamaModel的初始化
#         LMMMetaModel.__init__(self, config)  # 再调用LMMMetaModel的初始化
        
#         # 确保embed_tokens正确初始化并且维度正确
#         if self.embed_tokens is None:
#             self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
#     def forward(self, *args, **kwargs):
#         # 添加embedding层检查
#         if not isinstance(self.embed_tokens, nn.Embedding):
#             raise ValueError("embed_tokens is not properly initialized")
#         if not hasattr(self.embed_tokens, 'weight'):
#             raise ValueError("embed_tokens has no weight attribute")
#         return super().forward(*args, **kwargs)


# class LlamaLMMForCausalLM(LlamaForCausalLM, LMMMetaForCausalLM):
#     config_class = LlamaLMMConfig

#     def __init__(self, config):
#         LlamaForCausalLM.__init__(self, config)  # 先调用基类初始化
        
#         # 创建新的模型实例而不是重用现有的
#         self.model = LlamaLMMModel(config)
        
#         # 确保vocab_size和lm_head正确初始化
#         self.vocab_size = config.vocab_size
#         if not hasattr(self, 'lm_head') or self.lm_head is None:
#             self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
#         self.modalities = None
#         self._initialization_done = False
#         self._forward_setup_done = False
        
#         # 确保权重初始化
#         self.post_init()

#     def get_model(self) -> "LlamaLMMForCausalLM":
#         return self.model

#     def verify_model_if_needed(self):
#         """验证模型初始化状态"""
#         if not self._initialization_done:
#             try:
#                 if not hasattr(self.model, 'embed_tokens'):
#                     raise ValueError("Model has no embed_tokens")
#                 if not isinstance(self.model.embed_tokens, nn.Embedding):
#                     raise ValueError("embed_tokens is not nn.Embedding")
#                 if not hasattr(self.model.embed_tokens, 'weight'):
#                     raise ValueError("embed_tokens has no weight")
                    
#                 logging.info(f"Model vocabulary size: {self.vocab_size}")
#                 logging.info(f"Embedding layer exists with shape: {self.model.embed_tokens.weight.shape}")
                
#                 self._initialization_done = True
#             except Exception as e:
#                 logging.error(f"Model verification failed: {str(e)}")
#                 raise

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         # 在forward前验证模型状态
#         if not self._forward_setup_done:
#             self.verify_model_if_needed()
#             self._forward_setup_done = True

#         # 设置dtype和device
#         dtype = next(self.parameters()).dtype
#         device = next(self.parameters()).device

#         # 处理Flash Attention 2.0的dtype
#         if hasattr(self.config, 'use_flash_attention_2') and self.config.use_flash_attention_2:
#             if dtype not in [torch.float16, torch.bfloat16]:
#                 logging.warning("Flash Attention 2.0 requires torch.float16 or torch.bfloat16. Using autocast.")

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         training_mode = kwargs.get('training_mode', None)

#         try:
#             prepared_inputs = self.prepare_inputs_labels_for_multimodal(
#                 input_ids, attention_mask, past_key_values, labels, **kwargs
#             )
            
#             if len(prepared_inputs) == 5:
#                 input_ids, attention_mask, past_key_values, inputs_embeds, labels = prepared_inputs
#                 projected_tensors = None
#             else:
#                 input_ids, attention_mask, past_key_values, inputs_embeds, labels, projected_tensors = prepared_inputs
#         except Exception as e:
#             logging.error(f"Error in prepare_inputs_labels_for_multimodal: {str(e)}")
#             projected_tensors = None
#             inputs_embeds = None

#         # 处理输入
#         if input_ids is not None:
#             batch_size, sequence_length = input_ids.shape
            
#             if attention_mask is None:
#                 attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
#             elif attention_mask.dtype != torch.bool:
#                 attention_mask = attention_mask.bool()
                
#             if position_ids is None:
#                 position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

#             # 确保所有输入在正确的设备上
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             position_ids = position_ids.to(device)
#             if labels is not None:
#                 labels = labels.to(device)

#         # 使用autocast处理模型forward
#         with torch.cuda.amp.autocast(enabled=True):
#             outputs = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#             hidden_states = outputs[0]
#             logits = self.lm_head(hidden_states)
#             logits = logits.float()

#         # 计算损失
#         loss = None
#         if labels is not None and training_mode != "dpo":
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)
#             if projected_tensors is not None:
#                 loss = loss + projected_tensors.mean() * 0

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         past_key_values=None,
#         attention_mask=None,
#         inputs_embeds=None,
#         modality_inputs=None,
#         **kwargs
#     ):
#         """为生成准备输入"""
#         if past_key_values:
#             input_ids = input_ids[:, -1:]

#         if inputs_embeds is not None:
#             raise ValueError("inputs_embeds not supported")

#         # 构建模型输入
#         model_inputs = {
#             "input_ids": input_ids,
#             "position_ids": None,
#             "past_key_values": past_key_values,
#             "use_cache": kwargs.get("use_cache"),
#             "attention_mask": attention_mask,
#         }

#         # 添加模态输入（如果有）
#         if modality_inputs:
#             model_inputs.update(modality_inputs)

#         return model_inputs

#     def _reorder_cache(self, past_key_values, beam_idx):
#         """重新排序缓存用于beam search"""
#         if past_key_values is not None:
#             reordered_past = ()
#             for layer_past in past_key_values:
#                 reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) 
#                                        for past_state in layer_past),)
#             return reordered_past
#         return None

# from typing import List, Optional, Tuple, Union
# import logging
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from deepspeed import zero
# from deepspeed.runtime.zero import GatheredParameters

# from presto.language_models.base_model import LMMMetaModel, LMMMetaForCausalLM
# from typing import List, Optional, Tuple, Union
# import logging
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
# from transformers.modeling_outputs import CausalLMOutputWithPast
# try:
#     import deepspeed
#     from deepspeed import zero
#     from deepspeed.runtime.zero import GatheredParameters
#     DEEPSPEED_AVAILABLE = True
# except ImportError:
#     DEEPSPEED_AVAILABLE = False

# from presto.language_models.base_model import LMMMetaModel, LMMMetaForCausalLM

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

# try:
try:
    import deepspeed
    from deepspeed import zero
    from deepspeed.runtime.zero import GatheredParameters
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

        # 初始化 embed_tokens
        if not hasattr(self, "embed_tokens") or self.embed_tokens is None:
            self.embed_tokens = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id,
            )
            # 初始化权重
            self.embed_tokens.weight.data.normal_(mean=0.0, std=config.initializer_range)
            logging.info(f"Embedding layer initialized with shape: {self.embed_tokens.weight.shape}")
        else:
            logging.warning("Embedding layer already initialized.")

        # 验证 embed_tokens 权重形状
        if self.embed_tokens.weight.shape[0] <= 0 or self.embed_tokens.weight.shape[1] <= 0:
            raise ValueError(f"Embedding layer weight is not properly initialized! Current shape: {self.embed_tokens.weight.shape}")
    
class LlamaLMMForCausalLM(LlamaForCausalLM, LMMMetaForCausalLM):
    config_class = LlamaLMMConfig

    def __init__(self, config):
        # 初始化父类
        LlamaForCausalLM.__init__(self, config)

        # 避免 DeepSpeed 干扰
        # if DEEPSPEED_AVAILABLE:
        #     with zero.Init(enabled=False):  # 禁用 zero.Init
        #         self.model = LlamaLMMModel(config)
        #         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # else:
        self.model = LlamaLMMModel(config)
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        if labels is not None and not is_dpo:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if projected_tensors is not None:
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

# Register the model
AutoConfig.register("llama-lmm", LlamaLMMConfig)
AutoModelForCausalLM.register(LlamaLMMConfig, LlamaLMMForCausalLM)