#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"

MODEL_VERSION=phi-2
TRAIN_VERSION="sft-lora"
# EPOCH=$1
BASE_LLM_PATH="checkpoints/stage2/llava-moleculestm-phi-2-pretrain_all"
MODEL_LORA_PATH="/home/liyanhao/chemllm/REACT/checkpoints/sft/llava-moleculestm-phi-2-sft-notrans"

PROJECTOR_DIR="/home/liyanhao/chemllm/REACT/checkpoints/stage1/llava-moleculestm-phi-2-stage1"

DATA_DIR="/home/liyanhao/chemllm/REACT/datasets/MolInst_FS_125K_SMILES-MMChat"

# log path
LOG_DIR="./logs/lora/forward_reaction_prediction"

python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --model_lora_path $MODEL_LORA_PATH \
    --projectors_path $PROJECTOR_DIR \
    --lora_enable True \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose \