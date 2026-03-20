
#!/bin/bash

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)
export OMP_NUM_THREADS=16  # Adjust based on available CPU cores
export WANDB_DIR="/scratch/shared/beegfs/gabrijel/wandb"
export WORLD_SIZE=2
export CONFIG_PATH="/users/gabrijel/projects/vgg-wm-vae/configs"
export CONFIG_NAME="default_raw_cityscapes.yaml"
export BETA="0.1"

srun --export ALL \
  torchrun --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py model=raw/convnext_isotropic_base model.beta=${BETA} objective="VAE"