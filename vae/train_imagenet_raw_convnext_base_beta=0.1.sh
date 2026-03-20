#!/bin/bash

#SBATCH --job-name=train_imagenet_raw_convnext_base_beta=0.1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)
export OMP_NUM_THREADS=64  # Adjust based on available CPU cores
export WANDB_DIR="${SCRATCH}/wandb"
export WORLD_SIZE=4
export CONFIG_PATH="configs"
export CONFIG_NAME="default_raw_ilsvrc_isanbard.yaml"
export BETA="0.1"

srun --export=ALL torchrun \
  --nproc_per_node=$WORLD_SIZE \
  --standalone \
  train.py objective="VAE" model.beta=${BETA}
