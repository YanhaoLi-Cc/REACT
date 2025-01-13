#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"


MODEL_VERSION=vicuna-7b-v1.5
TRAIN_VERSION="sft-lora"
# EPOCH=$1

BASE_LLM_PATH="/home/liyanhao/chemllm/REACT/models/PRESTO"
MODEL_LORA_PATH="/home/liyanhao/chemllm/REACT/checkpoints/sft/llava-moleculestm-vicuna-7b-v1.5-sft-trans"

PROJECTOR_DIR="/home/liyanhao/chemllm/REACT/models/PRESTO/non_lora_trainables.bin"

DATA_DIR="/home/liyanhao/chemllm/REACT/datasets/MolInst_RS_125K_SMILES-MMChat"

# log path
LOG_DIR="/home/liyanhao/chemllm/REACT/sft_results/lora/retrosynthesis"

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