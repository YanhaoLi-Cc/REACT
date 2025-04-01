# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from presto.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import transformers
import logging

from presto.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from presto.data import DataArguments
from presto.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from presto.modalities import MODALITY_BUILDERS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    print(data_args)

    # 如果skip_modality_processing为True且未指定modality_builder，使用空模态列表
    if getattr(training_args, "skip_modality_processing", False) and not hasattr(model_args, "modality_builder"):
        logging.info("Skip modality processing enabled and no modality builder specified, using empty modality list")
        modalities = []
    # 如果跳过模态处理但提供了modality_builder，发出警告但仍使用空列表
    elif getattr(training_args, "skip_modality_processing", False):
        logging.warning("Skip modality processing enabled but modality builder specified, ignoring modality builder")
        modalities = []
    # 常规情况，使用指定的modality_builder
    else:
        modalities = MODALITY_BUILDERS[model_args.modality_builder]()
    
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_modalities(model_cls, training_args, model_args, modalities)
