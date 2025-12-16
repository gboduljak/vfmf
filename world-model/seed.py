

import logging
import random
import numpy as np
import torch

from typing import Optional

from utils import get_rank, rank_prefixed_message

log = logging.getLogger(__name__)


def seed_everything(
    seed: int,
    verbose: bool = True
) -> int:

  if verbose:
    log.info(rank_prefixed_message(f"Seed set to {seed}", get_rank()))

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  return seed


def worker_init_function(worker_id: int, global_rank: int) -> None:
  process_seed = torch.initial_seed()
  # back out the base seed so we can use all the bits
  base_seed = process_seed - worker_id
  log.debug(
      f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
  )
  seed_sequence = generate_seed_sequence(
      base_seed,
      worker_id,
      global_rank,
      count=4
  )
  torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
  # combine two 64-bit seeds
  random.seed((seed_sequence[1] << 32) | seed_sequence[2])
  ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
  np_rng_seed = ss.generate_state(4)
  np.random.seed(np_rng_seed)


def generate_seed_sequence(
    base_seed: int,
    worker_id: int,
    global_rank: int,
    count: int
) -> list[int]:
  """Generates a sequence of seeds from a base seed, worker id and rank using the linear congruential generator (LCG)
  algorithm."""
  # Combine base seed, worker id and rank into a unique 64-bit number
  combined_seed = (base_seed << 32) | (worker_id << 16) | global_rank
  seeds = []
  for _ in range(count):
    # x_(n+1) = (a * x_n + c) mod m. With c=1, m=2^64 and a is D. Knuth's
    # constant
    combined_seed = (
        combined_seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
    seeds.append(combined_seed)
  return seeds
