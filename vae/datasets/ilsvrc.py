import os
from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset


class ILSVRC(Dataset):
  def __init__(self, dataset_root: str, split: Literal["train", "val", "test"], transform=None):
    self.dataset_root = Path(dataset_root) / split / "images"
    self.transform = transform

    self.images = []
    for root, _, files in os.walk(self.dataset_root):
      for f in files:
        if f.lower().endswith('.png'):
          self.images.append(Path(root) / f)

    print(f"loaded {len(self.images)} samples from {dataset_root}")

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    img_path = self.images[idx]

    if self.transform:
      return self.transform([img_path]), img_path.stem

    return [img_path], img_path.stem
