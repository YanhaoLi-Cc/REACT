#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"
export WANDB_API_KEY="d82b836e8b00f02e07f039b2743a7896cbe0c3b9"

MODEL_VERSION=phi-2
BASE_LLM_PATH="/home/liyanhao/chemllm/REACT/checkpoints/stage2/llava-moleculestm-phi-2-pretrain_all_trans-new"
MODEL_CLS=PhiLMMForCausalLM

# output path
OUTPUT_DIR="checkpoints/sft/llava-moleculestm-$MODEL_VERSION-sft-trans"

# NUM_GPUS=8
# deepspeed --num_gpus=$NUM_GPUS scripts/train_model.py \
CUDA_VISIBLE_DEVICES=0 deepspeed scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --data_mixture "sft_subset" \
    --output_dir $OUTPUT_DIR \
    --lora_enable True \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
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
    --report_to none \
    --deepspeed configs/zero2.json