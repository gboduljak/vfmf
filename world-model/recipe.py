
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def sample_train_dino_foresight_variable_length(max_sequence_length: int):
  def sample(frames_path: List[Path]) -> List[Path]:
    sequence_length = max_sequence_length
    # [1,3] with equal probabilities 4 is excluded
    num_frames_skip = np.random.randint(0, 4)
    step = num_frames_skip + 1
    start_idx = np.random.randint(
        0,
        len(frames_path) - step * sequence_length + num_frames_skip + 1
    )
    return frames_path[start_idx: start_idx + step * sequence_length: step]

  return sample


def sample_train_dino_foresight(sequence_length: int):
  def sample(frames_path: List[Path]) -> List[Path]:
    # [1,3] with equal probabilities 4 is excluded
    num_frames_skip = np.random.randint(1, 4)
    step = num_frames_skip + 1
    start_idx = np.random.randint(
        0,
        len(frames_path) - step * sequence_length + num_frames_skip + 1
    )
    return frames_path[start_idx: start_idx + step * sequence_length: step]

  return sample


def sample_val_dino_foresight(sequence_length: int):
  def sample(frames_path: List[Path]) -> List[Path]:
    num_frames_skip = 2
    step = num_frames_skip + 1
    start_idx = 20 - step * sequence_length + num_frames_skip
    return frames_path[start_idx: start_idx + step * sequence_length: step]

  return sample


def load_frames(convert_to_rgb: bool = False):
  def load(frames_path: List[Path]) -> List[Image]:
    from PIL import Image
    if convert_to_rgb:
      return [Image.open(p).convert("RGB") for p in frames_path]
    else:
      return [Image.open(p) for p in frames_path]      

  return load
