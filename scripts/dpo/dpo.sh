#!/bin/bash

set -e  # 遇到错误时停止脚本执行

# 输出调试信息
echo "============================================"
echo "Starting DPO training script"
echo "Current directory: $(pwd)"
echo "============================================"

# 环境变量设置
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"
export WANDB_API_KEY="d82b836e8b00f02e07f039b2743a7896cbe0c3b9"
# export PYTHONPATH="$PYTHONPATH:$(pwd)"
# export CUDA_VISIBLE_DEVICES=2

# DeepSpeed调试环境变量
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 启用详细的分布式调试信息
export CUDA_LAUNCH_BLOCKING=1  # 避免异步CUDA操作，便于调试
export DS_SKIP_PARAM_CHECK=1  # 跳过DeepSpeed的部分参数检查
# 启用CUDA设备端断言，以便获得更详细的错误信息
export TORCH_USE_CUDA_DSA=1

MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH="/home/liyanhao/chemllm/REACT/models/PRESTO"
MODEL_CLS=LlamaLMMForCausalLM

# output path
OUTPUT_DIR="checkpoints/dpo/llava-moleculestm-$MODEL_VERSION-dpo-lora"

# DPO参数
DPO_BETA=0.1
MAX_PROMPT_LENGTH=64  # 减小提示长度以避免可能的越界问题
MAX_LENGTH=128  # 减小总长度以避免可能的越界问题
REF_MODEL=$BASE_LLM_PATH  # 使用相同的模型作为参考

# LoRA参数
LORA_RANK=2  # 减小rank降低内存使用
LORA_ALPHA=4
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# 执行完整训练
deepspeed --include localhost:0 scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --training_mode dpo \
    --data_mixture "dpo_mixture" \
    --output_dir $OUTPUT_DIR \
    --lora_enable True \
    --lora_r $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $TARGET_MODULES \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing False \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --model_max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --report_to wandb \
    --dpo_beta $DPO_BETA \
    --ref_model $REF_MODEL \
    --skip_modality_processing True \
    --offload_optimizer True \
    --offload_param True \
    --deepspeed configs/zero2.json

echo "Training completed!"