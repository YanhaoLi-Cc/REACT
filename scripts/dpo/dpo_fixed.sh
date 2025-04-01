#!/bin/bash

# 输出开始信息
echo "============================================"
echo "Starting DPO training script"
echo "Current directory: $(pwd)"
echo "============================================"

# 设置模型参数
MODEL_PATH="models/PRESTO"
MODEL_CLS="LlamaLMMForCausalLM"
TRAINING_MODE="dpo"
DATA_MIXTURE="dpo_mixture"
OUTPUT_DIR="checkpoints/dpo/llava-moleculestm-vicuna-7b-v1.5-dpo-lora"

# 设置 LoRA 参数
LORA_ENABLE="True"
LORA_R=4
LORA_ALPHA=8
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# 设置训练参数
BF16="True"
TF32="True"
NUM_EPOCHS=1
GRAD_CHECKPOINT="True"
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
MODEL_MAX_LENGTH=256
MAX_PROMPT_LENGTH=128
EVAL_STRATEGY="steps"
EVAL_STEPS=50
SAVE_STRATEGY="steps"
SAVE_STEPS=20
SAVE_TOTAL_LIMIT=3
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"
DATALOADER_WORKERS=2
LOGGING_STEPS=1
REPORT_TO="wandb"
DPO_BETA=0.1
REF_MODEL="models/PRESTO"
SKIP_MODALITY="True"
DEEPSPEED_CONFIG="configs/zero2.json"

# 设置要使用的 GPU 设备
GPU_DEVICE=2  # 使用 GPU 2

# 创建环境变量设置文件，解决各种问题
cat > fix_env.py << 'EOL'
import os
import sys
import logging

# 配置基本的日志级别
logging.basicConfig(level=logging.INFO)

def fix_environment():
    """设置必要的环境变量，解决各种训练问题"""
    
    # 解决梯度检查点问题
    os.environ["PYTORCH_CHECKPOINT_USE_REENTRANT"] = "0"
    
    # 解决 DeepSpeed 梯度检查问题
    os.environ["DS_SKIP_GRAD_REDUCE_CHECK"] = "1"
    
    # 解决 numactl 问题
    os.environ["DEEPSPEED_USE_CPU"] = "0"
    
    # 让 DeepSpeed 不使用 numactl
    os.environ["DEEPSPEED_NUMA_CONFIG"] = "disable"
    
    # 显示环境变量
    logging.info("Environment variables set:")
    for var in ["PYTORCH_CHECKPOINT_USE_REENTRANT", "DS_SKIP_GRAD_REDUCE_CHECK", 
                "DEEPSPEED_USE_CPU", "DEEPSPEED_NUMA_CONFIG"]:
        if var in os.environ:
            logging.info(f"  {var}={os.environ[var]}")
    
    # 尝试导入 PyTorch 并验证 CUDA 可用性
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logging.info(f"PyTorch found {device_count} CUDA devices")
            for i in range(device_count):
                logging.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            logging.warning("PyTorch CUDA is not available!")
    except ImportError:
        logging.warning("Could not import PyTorch")
    
    # 尝试修补梯度检查点
    try:
        sys.path.append(os.getcwd())
        from presto.checkpoint_utils import fix_deepspeed_checkpointing
        success = fix_deepspeed_checkpointing()
        if success:
            logging.info("Successfully patched gradient checkpointing")
        else:
            logging.warning("Failed to patch gradient checkpointing, but environment variables are set")
    except Exception as e:
        logging.error(f"Error importing checkpoint utils: {str(e)}")

if __name__ == "__main__":
    fix_environment()
    logging.info("Environment setup complete")
EOL

# 运行环境修复脚本
echo "Setting up environment variables and patches..."
python fix_env.py

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 启动训练
echo "Starting DPO training with fixed configuration..."
deepspeed --include localhost:0 scripts/train_model.py \
    --model_name_or_path $MODEL_PATH \
    --model_cls $MODEL_CLS \
    --training_mode $TRAINING_MODE \
    --data_mixture $DATA_MIXTURE \
    --output_dir $OUTPUT_DIR \
    --lora_enable $LORA_ENABLE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --bf16 $BF16 \
    --tf32 $TF32 \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_checkpointing $GRAD_CHECKPOINT \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --model_max_length $MODEL_MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --evaluation_strategy $EVAL_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER \
    --dataloader_num_workers $DATALOADER_WORKERS \
    --logging_steps $LOGGING_STEPS \
    --report_to $REPORT_TO \
    --dpo_beta $DPO_BETA \
    --ref_model $REF_MODEL \
    --skip_modality_processing $SKIP_MODALITY \
    --deepspeed $DEEPSPEED_CONFIG 