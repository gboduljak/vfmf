
from pathlib import Path
from typing import List

import numpy as np
from PIL.Image import Image


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


def sample_frame():
  def sample(frames_path: List[Path]) -> List[Path]:
    # [1,3] with equal probabilities 4 is excluded
    frame_idx = np.random.randint(0, len(frames_path))
    return [frames_path[frame_idx]]

  return sample


def load_frames():
  def load(frames_path: List[Path]) -> List[Image]:
    from PIL import Image
    return [Image.open(p) for p in frames_path]

  return load


def sample_frames(num_frames: int):
  def sample(frames_path: List[Path]) -> List[Path]:
    n = len(frames_path)
    if n == 0:
      return []
    if n <= num_frames:
      # Not enough frames, take all and pad with the last one
      frames = frames_path + [frames_path[-1]] * (num_frames - n)
      return frames[:num_frames]

    # Choose start index so that the window fits fully
    start = np.random.randint(0, n - num_frames + 1)
    end = start + num_frames
    return frames_path[start:end]

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
