import os
import random
from pathlib import Path
from typing import List, TypeVar

from PIL import Image
from torch.utils.data import Dataset

A = TypeVar('A')  # Generic type


def deterministic_shuffle(lst: List[A], seed: int = 42) -> List[A]:  # type: ignore
  """
  Returns a deterministically shuffled version of lst using the given seed.
  The original list is not modified.
  """
  shuffled = lst.copy()
  rng = random.Random(seed)  # Create a separate random generator
  rng.shuffle(shuffled)
  return shuffled

# class Cityscapes(Dataset):
#   def __init__(self, dataset_root, split='train', transform=None, benchmark=False):
#     self.root = dataset_root
#     self.split = split
#     self.transform = transform

#     self.cities = []
#     self.sequence_to_frames = {}
#     self.num_sequences = 0

#     base_dir = os.path.join(dataset_root, 'leftImg8bit_sequence', split)

#     for city in sorted(os.listdir(base_dir)):
#       city_dir = os.path.join(base_dir, city)
#       self.cities.append(city)

#       for frame_name in sorted(os.listdir(city_dir)):
#         if not frame_name.endswith('.png'):
#           continue

#         # Expected pattern: city_sequence_frame_leftImg8bit.png
#         parts = frame_name.split('_')
#         if len(parts) < 4:
#           raise ValueError(
#               f"Unexpected filename format: '{frame_name}' "
#               f"in city directory '{city_dir}'"
#           )

#         city_name, sequence_id, frame_idx, *_ = parts
#         frame_idx = int(frame_idx)
#         frame_path = os.path.join(city_dir, frame_name)

#         sequence_key = f"{city_name}_{sequence_id}"

#         if sequence_key not in self.sequence_to_frames:
#           self.sequence_to_frames[sequence_key] = []

#         self.sequence_to_frames[sequence_key].append(
#             (frame_idx, frame_path)
#         )

#     # Sort frames within each sequence by frame index
#     for seq in self.sequence_to_frames:
#       self.sequence_to_frames[seq] = [
#           path for _, path in sorted(self.sequence_to_frames[seq])
#       ]

#     # List of all unique sequence IDs
#     self.sequences = sorted(self.sequence_to_frames.keys())
#     self.num_sequences = len(self.sequences)

#   def __len__(self):
#     return self.num_sequences

#   def __getitem__(self, idx):
#     seq_key = self.sequences[idx]
#     frame_paths = self.sequence_to_frames[seq_key]
#     if self.transform:
#       x = self.transform(frame_paths)
#       return x, seq_key
#     else:
#       return [Image.open(p).convert('RGB') for p in frame_paths], seq_key


class Cityscapes(Dataset):
  def __init__(self, dataset_root, split='train', transform=None, benchmark=False):
    self.root = dataset_root
    self.split = split
    self.transform = transform

    self.cities = []
    self.sequence_to_frames = {}
    self.num_sequences = 0

    base_dir = os.path.join(dataset_root, 'leftImg8bit_sequence', split)

    for city in sorted(os.listdir(base_dir)):
      city_dir = os.path.join(base_dir, city)
      self.cities.append(city)

      for frame_name in sorted(os.listdir(city_dir)):
        if not frame_name.endswith('.png'):
          continue

        # Expected pattern: city_sequence_frame_leftImg8bit.png
        parts = frame_name.split('_')
        if len(parts) < 4:
          raise ValueError(
              f"Unexpected filename format: '{frame_name}' "
              f"in city directory '{city_dir}'"
          )

        city_name, sequence_id, frame_idx, *_ = parts
        frame_idx = int(frame_idx)
        frame_path = os.path.join(city_dir, frame_name)

        sequence_key = f"{city_name}_{sequence_id}"

        if sequence_key not in self.sequence_to_frames:
          self.sequence_to_frames[sequence_key] = []

        self.sequence_to_frames[sequence_key].append((frame_idx, frame_path))

    # Sort frames within each sequence by frame index
    for seq in list(self.sequence_to_frames.keys()):
      frames = [path for _, path in sorted(self.sequence_to_frames[seq])]
      if len(frames) == 30:
        self.sequence_to_frames[seq] = frames
      else:
        del self.sequence_to_frames[seq]

    # List of all unique sequence IDs (after filtering)
    self.sequences = sorted(self.sequence_to_frames.keys())
    self.num_sequences = len(self.sequences)

  def __len__(self):
    return self.num_sequences

  def __getitem__(self, idx):
    seq_key = self.sequences[idx]
    frame_paths = self.sequence_to_frames[seq_key]
    if self.transform:
      x = self.transform(frame_paths)
      return x, seq_key
    else:
      return [Image.open(p).convert('RGB') for p in frame_paths], seq_key


class CityscapesFrameDataset(Dataset):
  def __init__(self, dataset_root, split='train', transform=None, benchmark=False, size_limit=None):
    self.root = dataset_root
    self.split = split
    self.transform = transform
    if benchmark:
      seq_dir = 'leftImg8bit'
    else:
      seq_dir = 'leftImg8bit_sequence'
    base_dir = os.path.join(dataset_root, seq_dir, split)
    self.frames = []

    for city in sorted(os.listdir(base_dir)):
      city_dir = os.path.join(base_dir, city)
      for frame_name in sorted(os.listdir(city_dir)):
        if frame_name.endswith('.png'):
          frame_path = os.path.join(city_dir, frame_name)
          self.frames.append(frame_path)

    self.frames = list(sorted(self.frames))
    if size_limit is not None:
      self.frames = deterministic_shuffle(self.frames)
      self.frames = self.frames[:size_limit]
    print(f"Loaded {len(self.frames)} frames from split '{split}'")

  def __len__(self):
    return len(self.frames)

  def __getitem__(self, idx):
    img_path = self.frames[idx]

    if self.transform:
      return (
          self.transform([img_path]),
          Path(img_path).stem
      )
    else:
      return (
          Image.open(img_path),
          Path(img_path).stem
      )
