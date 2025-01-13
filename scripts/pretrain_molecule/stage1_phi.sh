#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"
export WANDB_API_KEY="d82b836e8b00f02e07f039b2743a7896cbe0c3b9"

MODEL_VERSION=phi-2
BASE_LLM_PATH=/home/liyanhao/chemllm/REACT/models/$MODEL_VERSION
MODEL_CLS=PhiLMMForCausalLM
DATA_DIR="/home/liyanhao/chemllm/REACT/datasets/"
OUTPUT_DIR="checkpoints/stage1/llava-moleculestm-$MODEL_VERSION-stage1"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --dataset_path $DATA_DIR \
    --data_mixture "pubchem_cap" \
    --output_dir $OUTPUT_DIR \
    --pretrain_projectors \
    --lora_enable False \
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
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 8 \
    --logging_steps 1 \
    --report_to wandb \
    --deepspeed configs/zero1.json