
import os

import torch
import torch.distributed as dist

from utils import rank_prefixed_message


def setup_ddp():
  """Initializes the distributed process group."""
  backend = 'nccl'

  if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(
        f"Running via torchrun: RANK {rank}, WORLD_SIZE {world_size}, LOCAL_RANK {local_rank}")
  elif 'SLURM_NTASKS' in os.environ:
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = rank % torch.cuda.device_count()
    print(
        f"Running via SLURM: RANK {rank}, WORLD_SIZE {world_size}, LOCAL_RANK {local_rank}")
  else:
    raise RuntimeError("Could not find DDP environment variables.")

  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(local_rank)
  print(rank_prefixed_message(f"Set device to cuda:{local_rank}", rank=rank))

  return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def cleanup():
  """Cleans up the distributed process group."""
  dist.destroy_process_group()
