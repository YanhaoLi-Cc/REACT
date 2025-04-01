from typing import List, Dict
from abc import ABC, abstractmethod

from torch.nn.functional import conv1d
import torch
import logging

from presto.modalities.base_modality import Modality

class LMMMetaModel:
    def __init__(self, config):
        super(LMMMetaModel, self).__init__(config)

    def _load_projector_weights(self, weights: Dict):
        weights = {k.replace("base_model.", "").replace("model.", ""): v for k, v in weights.items()}
        logging.info(f"Loading pretrained weights: {list(weights.keys())}")
        load_result = self.load_state_dict(weights, strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), "Unexpected weights, is this the right model?"

    def initialize_pretrained_modules(self, modalities: List[Modality], weights: Dict):
        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)

        self._load_projector_weights(weights)

    def initialize_modules(self, modalities: List[Modality], weights: Dict):
        names = [m.name for m in modalities]

        self.config.modalities = names

        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)

        self._load_projector_weights(weights)


class LMMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> "LMMMetaForCausalLM":
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        model = self.get_model()
        
        # 确保input_ids不为None
        if input_ids is None:
            return None, attention_mask, past_key_values, None, labels
            
        # 检查输入有效性
        try:
            batch_size, seq_len = input_ids.shape
        except Exception as e:
            logging.error(f"Invalid input_ids shape: {e}")
            # 如果input_ids无效，返回原始输入
            return input_ids, attention_mask, past_key_values, None, labels
            
        # DPO模式检查
        is_dpo = kwargs.get('training_mode') == 'dpo'
        if is_dpo:
            batch_size = batch_size // 2
        
        # 验证batch_size
        if batch_size <= 0:
            logging.error(f"Invalid batch_size: {batch_size}")
            # 返回原始输入，避免后续处理
            return input_ids, attention_mask, past_key_values, None, labels

        # 初始化embeddings
        try:
            inputs_embeds = torch.zeros(
                (batch_size * (2 if is_dpo else 1), seq_len, self.config.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        except Exception as e:
            logging.error(f"Error initializing inputs_embeds: {e}")
            # 如果无法初始化嵌入，返回原始输入
            return input_ids, attention_mask, past_key_values, None, labels

        # 检查是否需要处理模态
        has_modality = False
        if hasattr(self, 'modalities') and self.modalities is not None:
            for m in self.modalities:
                modal_data = kwargs.get(m.name)
                if modal_data is not None:
                    # 检查模态数据的有效性
                    if isinstance(modal_data, list) and len(modal_data) > 0:
                        has_modality = True
                        break
        
        if not has_modality:
            return input_ids, attention_mask, past_key_values, None, labels

        # 处理模态
        projected_tensors = []
        if past_key_values is None:
            for m in self.modalities:
                try:
                    modal_data = kwargs.get(m.name)
                    if modal_data is None:
                        continue  # 跳过没有数据的模态
                        
                    m_vals = m.forward(modal_data)
                    if not m_vals:
                        continue  # 跳过空的模态数据
                        
                    mp_vals = []
                    proj = getattr(model, m.name + "_lmm_projector", None)
                    if proj is None:
                        logging.error(f"Projector for modality {m.name} not found")
                        continue
                        
                    for m_val in m_vals:
                        instance_val_list = []
                        for each_instance in m_val:
                            if each_instance is not None:
                                instance_val = proj(each_instance)
                            else:
                                # 使用与模型匹配的设备和数据类型
                                each_instance = torch.zeros((1, 300), device=self.device, dtype=self.dtype)
                                instance_val = proj(each_instance)
                            instance_val_list.append(instance_val)
                        mp_vals.append(instance_val_list)
                        
                    projected_tensors.append(mp_vals)
                except Exception as e:
                    logging.error(f"Error processing modality {m.name}: {e}")
                    # 继续处理下一个模态，不中断整个流程
            
            # 如果没有任何有效的模态处理成功，返回原始输入
            if not projected_tensors:
                return input_ids, attention_mask, past_key_values, None, labels

        # 处理embeddings
        try:
            for i, input_ids_sample in enumerate(input_ids):
                is_text_mask = input_ids_sample >= 0
                inputs_embeds[i, is_text_mask] = model.embed_tokens(input_ids_sample[is_text_mask])

                if is_text_mask.sum() == seq_len:
                    continue

                if past_key_values is not None:
                    logging.warning("Cannot have cached keys during instruction pass")
                    continue

                for mi, m in enumerate(self.modalities):
                    if mi >= len(projected_tensors):
                        continue  # 跳过未处理成功的模态
                        
                    m_mask = (input_ids_sample == m.token_idx).float()
                    if m_mask.sum() == 0:
                        continue  # 没有这个模态的token
                
                    try:
                        instances_token_width = [instance.shape[0] for instance in projected_tensors[mi][i]]
                        indices = []
                        ii = 0
                        while ii < len(m_mask):
                            if m_mask[ii] == 1:
                                if len(indices) >= len(instances_token_width):
                                    break
                                indices.append(ii)
                                ii += instances_token_width[len(indices) - 1]
                            else:
                                ii += 1
                            if len(indices) == len(instances_token_width):
                                break

                        last_covered_idx = -1
                        for k, possible_token_idx in enumerate(indices):
                            if possible_token_idx <= last_covered_idx or k >= len(instances_token_width):
                                continue
                                
                            if i >= len(projected_tensors[mi]) or k >= len(projected_tensors[mi][i]):
                                continue  # 索引超出范围，跳过
                                
                            batch_modality_tensor = projected_tensors[mi][i][k]
                            width = instances_token_width[k]
                            
                            # 确保不超出序列长度
                            if possible_token_idx + width > seq_len:
                                width = seq_len - possible_token_idx
                                
                            if width > 0:
                                inputs_embeds[i, possible_token_idx:possible_token_idx + width] = batch_modality_tensor[:width]
                                last_covered_idx = possible_token_idx + width - 1

                    except Exception as e:
                        logging.error(f"Error processing embeddings for modality {m.name}: {e}")
                        # 继续处理，不中断整个流程
        except Exception as e:
            logging.error(f"Error processing embeddings: {e}")
            # 如果嵌入处理失败，返回原始输入
            return input_ids, attention_mask, past_key_values, None, labels

        # 清理和整理输出
        input_ids_copy = input_ids  # 保存一个副本，以备后面处理失败时使用
        del input_ids
        
        try:
            if projected_tensors:
                # 尝试更安全的方式处理projected_tensors
                try:
                    stacked_tensors = []
                    for modality in projected_tensors:
                        modality_tensors = []
                        for batch in modality:
                            try:
                                # 确保batch中的所有tensor都是有效的
                                if all(tensor is not None for tensor in batch):
                                    batch_tensor = torch.cat(batch, dim=0)
                                    modality_tensors.append(batch_tensor)
                            except Exception as e:
                                logging.warning(f"Failed to concatenate batch tensors: {e}")
                        
                        if modality_tensors:
                            try:
                                modality_tensor = torch.cat(modality_tensors, dim=0)
                                stacked_tensors.append(modality_tensor)
                            except Exception as e:
                                logging.warning(f"Failed to concatenate modality tensors: {e}")
                    
                    if stacked_tensors:
                        projected_tensors = torch.stack(stacked_tensors)
                    else:
                        projected_tensors = None
                except Exception as e:
                    logging.error(f"Error stacking projected tensors: {e}")
                    projected_tensors = None
            else:
                projected_tensors = None
        except Exception as e:
            logging.error(f"Error processing final projected tensors: {e}")
            projected_tensors = None
            # 如果处理过程失败，返回原始输入
            return input_ids_copy, attention_mask, past_key_values, None, labels

        return None, attention_mask, past_key_values, inputs_embeds, labels, projected_tensors