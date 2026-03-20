
import os
from typing import Callable, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class KubricDataset(Dataset):
  def __init__(
      self,
      dataset_root: str,
      split="train",
      suite="movi_a",
      original_resolution=256,
      transform: Callable[
          [List[Image.Image]],
          torch.Tensor
      ] = lambda x: x  # type: ignore
  ):
    super().__init__()
    self.data_path = dataset_root
    self.transform = transform
    self.sequences = []
    self.sequence_to_frames = {}
    seqs_dir = os.path.join(
        dataset_root,
        suite,
        f'{original_resolution}x{original_resolution}',
        split
    )
    self.num_frames = 0
    for seq in os.listdir(seqs_dir):
      frames = list(
          sorted([
              os.path.join(seqs_dir, seq, f)
              for f in os.listdir(os.path.join(seqs_dir, seq))
              if f.endswith("_rgb.png")
          ])
      )
      self.sequences.append(seq)
      self.sequence_to_frames[seq] = frames
      self.num_frames += len(frames)
    self.sequences = list(sorted(self.sequences))

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence_name = self.sequences[idx]
    frames_paths = self.sequence_to_frames[sequence_name]

    if self.transform:
      x = self.transform(frames_paths)
      return x, sequence_name
    else:
      frames = [
          Image.open(frame).convert('RGB')
          for frame in frames_paths
      ]
      return frames, sequence_name
