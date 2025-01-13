#!/bin/bash

## We utilize 'Interleaved Data' + 'Name Conversion Data' + 'graph-to-smiles Data'

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"

MODEL_VERSION=vicuna-7b-v1.5
BASE_LLM_PATH=/home/liyanhao/chemllm/REACT/models/$MODEL_VERSION
MODEL_CLS=LlamaLMMForCausalLM
# output path
PRETRAIN_VERSION="pretrain_all"
OUTPUT_DIR="checkpoints/stage2/llava-moleculestm-$MODEL_VERSION-$PRETRAIN_VERSION"
# load stage-1 projector
PROJECTOR_DIR="checkpoints/stage1/llava-moleculestm-$MODEL_VERSION-stage1/lmm_projector.bin"

# NUM_GPUS=4
# deepspeed --num_gpus=$NUM_GPUS scripts/train_model.py \

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # 如果你的设备不支持 InfiniBand，可禁用它
# export NCCL_P2P_LEVEL=SYS
# export NCCL_SHM_DISABLE=1
# export NCCL_ASYNC_ERROR_HANDLING=1


CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed scripts/train_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_cls $MODEL_CLS \
    --modality_builder molecule_2d \
    --data_mixture "pretrain_v3" \
    --output_dir $OUTPUT_DIR \
    --pretrained_projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --report_to none \
    --deepspeed configs/zero3_offload.json