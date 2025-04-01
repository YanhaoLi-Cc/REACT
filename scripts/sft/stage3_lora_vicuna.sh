#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"
export WANDB_API_KEY="d82b836e8b00f02e07f039b2743a7896cbe0c3b9"

MODEL_VERSION=vicuna-7b-v1.5

BASE_LLM_PATH="/home/liyanhao/chemllm/REACT/models/PRESTO"

MODEL_CLS=LlamaLMMForCausalLM

# output path
OUTPUT_DIR="checkpoints/sft/llava-moleculestm-$MODEL_VERSION-sft-trans"

deepspeed --include localhost:2,3 scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --data_mixture "sft_subset_01" \
    --output_dir $OUTPUT_DIR \
    --lora_enable True \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --report_to wandb \
    --deepspeed configs/zero2.json
