import os
from typing import Optional


def get_rank() -> Optional[int]:
  # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
  # therefore LOCAL_RANK needs to be checked first
  rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
  for key in rank_keys:
    rank = os.environ.get(key)
    if rank is not None:
      return int(rank)
  # None to differentiate whether an environment variable was set at all
  return None


def rank_prefixed_message(message: str, rank: Optional[int]) -> str:
  if rank is not None:
    return f"[rank: {rank}] {message}"
  return message
