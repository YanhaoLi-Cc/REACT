## Installation

1. Install the required dependencies:
   ```
   conda create -n presto python=3.10
   pip install -r requirements.txt
   pip install -e .
   ```

2. Wget MoleculeSTM & vicuna
   ```
   mkdir models
   cd models

   mkdir MoleculeSTM
   wget https://huggingface.co/chao1224/MoleculeSTM/resolve/main/demo/demo_checkpoints_Graph/molecule_model.pth -P MoleculeSTM
   cd ..

   git lfs install 
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/lmsys/vicuna-7b-v1.5
   git lfs pull
   ```

3. Set up the necessary environment variables:
   ```
   export MOLECULE_2D_PATH="/path/to/MoleculeSTM/"
   export WANDB_API_KEY="your_wandb_api_key"
   ```

4. Prepare data
   ```
   mkdir datasets
   git lfs install
   git clone https://huggingface.co/datasets/OpenMol/PubChem_G2S_300K_SMILES-MMPretrain
   git clone https://huggingface.co/datasets/OpenMol/USPTO_RXN_Interleaved
   git clone https://huggingface.co/datasets/OpenMol/BH-SM_YR_10K-MMChat
   git clone https://huggingface.co/datasets/OpenMol/MolInst_FS_125K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/MolInst_RS_125K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/MolInst_FS_125K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/HTE_RAS_4K-MMChat
   git clone https://huggingface.co/datasets/OpenMol/RCR_RP_57K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/RCR_SP_70K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/RCR_CP_10K_SMILES-MMChat
   git clone https://huggingface.co/datasets/OpenMol/SMol_S2F_270K-MMChat
   git clone https://huggingface.co/datasets/OpenMol/SMol_S2I_270K-MMChat
   git clone https://huggingface.co/datasets/OpenMol/SMol_I2S_270K-MMChat
   git clone https://huggingface.co/datasets/OpenMol/SMol_I2F_270K-MMChat
   ```

## Pretraining

### Stage 1: Molecule-Text Alignment

To perform Stage 1 pretraining for molecule-text alignment, run the following command:
```bash
bash scripts/pretrain_multi_molecule/stage1.sh
```

This script will pre-train the model using the PubChem caption dataset and save the pretrained model checkpoints.

### Stage 2: Domain Incremental Pretraining

For Stage 2 pretraining, there are several configurations available:

- `stage2.sh`: Pretraining using interleaved molecule-text data from USPTO-Application.
- `stage2_rxn_nc.sh`: Pretraining using interleaved reaction data and name conversion tasks (g2s, s(g)2i, s(g)2f).
- `stage2_all.sh`: Pretraining using interleaved reaction data and all name conversion tasks (i2s, i2f).
- `stage2_skip_align.sh`: Skipping Stage 1 and directly starting with Stage 2 pretraining, only training the projector.
- `stage2_skip_align_fulltune.sh`: Skipping Stage 1 and directly starting with Stage 2 pretraining, finetuning the entire model.

To run a specific Stage 2 pretraining configuration, execute the corresponding script. For example:
```bash
bash scripts/pretrain_multi_molecule/stage2_rxn_nc.sh
```

## SFT (Stage 3) Downstream Tasks

For Stage 3 finetuning, we include finetuning scripts for various downstream tasks. Each task has its own directory under `scripts/build_dataset/` to build the dataset and `scripts/sft/` to run the finetuning. There are several configurations available:

- `stage3_freezeLLM.sh`: Finetuning the projector with a frozen LLM on Stage 3 downstream tasks.
- `stage3_lora.sh`: Finetuning the projector and applying LoRA to train the LLM on Stage 3 downstream tasks.
- `stage3_rxn_nc.sh`: Finetuning the LLM (pretrained using `stage2_rxn_nc.sh`) on Stage 3 downstream tasks.
- `stage3_skip_align_fulltune.sh`: Skipping Stage 1 and training with the full model on Stage 2 pretraining data and Stage 3 downstream tasks.
- `stage3_skip_stage2.sh`: Skipping Stage 2 and training with the full model on Stage 1 pretraining data and Stage 3 downstream tasks.
- `stage3_skip_stage12.sh`: Skipping Stage 1 and 2 and training with the full model on Stage 3 downstream tasks.
- `stage3.sh`: Train with the full model on Stage 3 directly.

To run a specific Stage 3 finetuning configuration, execute the corresponding script. For example:
```bash
bash scripts/sft/sft_lora/stage3_rxn_nc.sh $EPOCH $MODEL_VERSION
# $EPOCH: the epoch number to finetune the model (e.g., 3)
# $MODEL_VERSION: the model version to finetune (e.g., SFT-ALL)
```

## Evaluation

Here is a list of all the downstream tasks and the corresponding commands to run the evaluation:

### Reaction Prediction
#### Forward Prediction

To evaluate the forward reaction prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_forward_reaction_prediction.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_forward_reaction_prediction.sh $EPOCH $MODEL_VERSION
```

#### Retrosynthesis Prediction

To evaluate the retrosynthesis prediction task, use the following command:
```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_retrosynthesis.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_retrosynthesis.sh $EPOCH $MODEL_VERSION
```

### Reaction Condition Prediction
#### Reagent Prediction
To evaluate the reagent prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reagent_prediction.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_reagent_prediction.sh $EPOCH $MODEL_VERSION
```

#### Catalyst Prediction
To evaluate the catalyst prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_catalyst_prediction.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_catalyst_prediction.sh $EPOCH $MODEL_VERSION
```

#### Solvent Prediction
To evaluate the solvent prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_solvent_prediction.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_solvent_prediction.sh $EPOCH $MODEL_VERSION
```

### Reaction Condition Recommendation  
#### Reagent Selection
To evaluate the reagent selection task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reagent_selection.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_reagent_selection.sh $EPOCH $MODEL_VERSION
```

### Reaction Type Classification
To evaluate the reaction type classification task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_reaction_classification.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_reaction_classification.sh $EPOCH $MODEL_VERSION
```

### Yield Prediction
To evaluate the yield prediction task, use the following commands:

```bash
# For lora model
bash scripts/evaluate/sft_lora/evaluate_yields_regression.sh $EPOCH $MODEL_VERSION

# For full model
bash scripts/evaluate/sft_full/evaluate_yields_regression.sh $EPOCH $MODEL_VERSION
```

## Model Serving

To serve the trained model using a Flask server, run:

```
python scripts/serve_model.py --model_name_or_path <path_to_model> --model_lora_path <path_to_lora_model> --port <port_number>
```

This will start a Flask server that exposes a `/generate` endpoint for generating predictions using the trained model.

## Dataset Preparation

The `scripts/build_dataset` directory contains scripts for preparing datasets for different tasks. To prepare the datasets, follow the instructions within each task-specific directory.

- NOTE: Huggingface Dataset under preparation. Once the dataset is ready, we will sync the readme.
```
