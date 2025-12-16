import glob
import os
import os.path as osp
import random
from pathlib import Path
from typing import List, Optional, TypeVar

from PIL import Image
from torch.utils.data import Dataset

T = TypeVar('T')  # Generic type


def deterministic_shuffle(lst: List[T], seed: int = 42) -> List[T]:
  """
  Returns a deterministically shuffled version of lst using the given seed.
  The original list is not modified.
  """
  shuffled = lst.copy()
  rng = random.Random(seed)  # Create a separate random generator
  rng.shuffle(shuffled)
  return shuffled


class Cityscapes(Dataset):
  def __init__(
      self,
      dataset_root,
      split='train',
      transform=None,
  ):
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

        self.sequence_to_frames[sequence_key].append(frame_path)

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


class CityscapesBenchmarkDataset(Dataset):

  def __init__(
      self,
      dataset_root,
      split='train',
      transform=None,
      take: Optional[int] = None
  ):
    super().__init__()
    self.data_path = dataset_root
    self.split = split
    self.transform = transform

    self.num_frames = len(
        glob.glob(osp.join(self.data_path, split, '**', "*.png"))
    )

    self.sequences = set()
    for city_folder in glob.glob(osp.join(self.data_path, split, '*')):
      city_name = osp.basename(city_folder)
      frames_in_city = glob.glob(osp.join(city_folder, '*'))
      city_seqs = set(
          [
              f"{city_name}_{osp.basename(frame).split('_')[1]}"
              for frame in frames_in_city
          ]
      )
      # Note that in some cities very few, though very long sequences were
      # recorded
      if len(city_seqs) < 10:
        for seq in city_seqs:
          sub_seqs = sorted(
              glob.glob(
                  osp.join(
                      self.data_path,
                      split,
                      city_name,
                      seq + '*.png'))
          )
          sub_seq_startframe_ids = [
              osp.basename(
                  sub_seqs[i])[
                  :-
                  16] for i in range(
                  len(sub_seqs)) if i %
              30 == 0]
          self.sequences.update(sub_seq_startframe_ids)
      else:
        self.sequences.update(city_seqs)
        continue

    self.sequences = list(sorted(self.sequences))
    if take is not None:
      self.sequences = deterministic_shuffle(self.sequences)
      self.sequences = self.sequences[:take]
    self.sequences = list(sorted(self.sequences))

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence_name = self.sequences[idx]  # type: ignore
    splits = sequence_name.split("_")
    if len(splits) == 2:  # Sample from Short sequences
      frames_paths = sorted(
          # Sequence of 30 frames
          glob.glob(
              osp.join(
                  self.data_path,
                  self.split,
                  splits[0],
                  sequence_name +
                  '*.png')
          )
      )
    else:  # Sample from Long sequences
      frames_paths = [
          osp.join(
              self.data_path,
              self.split,
              splits[0],
              splits[0] +
              "_" +
              splits[1] +
              "_" +
              '{:06d}'.format(
                  int(
                      splits[2]) +
                  i) +
              '_leftImg8bit.png')
          for i in range(30)
      ]
    if self.transform:
      x = self.transform(frames_paths)
      return x, sequence_name
    else:
      return (
          [Image.open(p).convert('RGB') for p in frames_paths],
          sequence_name
      )


class CityscapesFrameDataset(Dataset):
  def __init__(self, dataset_root, split='train',
               transform=None, benchmark=False):
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
