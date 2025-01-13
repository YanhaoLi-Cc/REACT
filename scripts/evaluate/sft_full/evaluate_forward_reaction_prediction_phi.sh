#!/bin/bash

# set as environment variables
export MOLECULE_2D_PATH="/home/liyanhao/chemllm/REACT/models/MoleculeSTM"
export HF_HOME="/home/liyanhao/huggingface_cache"

TASK=forward_reaction_prediction

# MODEL_VERSION=phi-2
MODEL_VERSION=vicuna-7b-v1.5
# EPOCH=$1
# TRAIN_VERSION=$2 # e.g. full_all

DATA_DIR="/home/liyanhao/chemllm/REACT/datasets/MolInst_FS_125K_SMILES-MMChat"
BASE_LLM_PATH="/home/liyanhao/chemllm/REACT/models/PRESTO"
PROJECTOR_DIR="$BASE_LLM_PATH/non_lora_trainables.bin"

# log path
LOG_DIR="./logs/full/$TASK/epoch-$EPOCH"

python scripts/evaluate_model.py \
    --model_name_or_path $BASE_LLM_PATH \
    --projectors_path $PROJECTOR_DIR \
    --lora_enable False \
    --dataset_path  $DATA_DIR \
    --max_new_tokens 256 \
    --cache_dir $LOG_DIR \
    --output_dir $LOG_DIR \
    --evaluator "smiles" \
    --verbose \