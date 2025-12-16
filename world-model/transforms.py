
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


def normalize_dino():
  # params for DINO
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  return T.Normalize(mean=mean, std=std)


def denormalize_dino():
  # params for DINO
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)

  inv_mean = [-m / s for m, s in zip(mean, std)]
  inv_std = [1 / s for s in std]

  return T.Normalize(mean=inv_mean, std=inv_std)


def transform_train(
    resize_size: Tuple[int, int],
    random_horizontal_flip: bool = True,
    random_crop: bool = True,
    dinov2: bool = True,
):
  def identity(x): return x

  # Random crop (applied on tensors after conversion)
  if random_crop:
    def maybe_random_crop(frames: List[Image]) -> List[Image]:
      if np.random.rand() > 0.5:
        H, W = frames[0].height, frames[0].width
        s_f = np.random.rand() / 2 + 0.5  # random scale in [0.5, 1.0]
        crop_h, crop_w = int(H * s_f), int(W * s_f)
        i, j, h, w = T.RandomCrop.get_params(
            frames[0],  # type: ignore
            output_size=(crop_h, crop_w)
        )
        return [F.crop(f, i, j, h, w) for f in frames]  # type: ignore
      return frames
  else:
    maybe_random_crop = identity  # type: ignore

  # Horizontal flip
  if random_horizontal_flip:
    def maybe_horizontal_flip(
        frames: List[torch.Tensor]
    ) -> List[torch.Tensor]:
      if np.random.rand() > 0.5:
        return [F.hflip(f) for f in frames]
      return frames
  else:
    maybe_horizontal_flip = identity  # type: ignore

  # Basic transforms applied to individual frames
  base_transform = T.Compose([
      T.Resize(resize_size, antialias=True),
      T.ToTensor(),
      normalize_dino() if dinov2 else identity
  ])

  def transform(frames: List[Image]) -> torch.Tensor:
    x = frames
    x = maybe_random_crop(x)  # type: ignore
    x = maybe_horizontal_flip(x)  # type: ignore
    x = [base_transform(f) for f in frames]
    x = torch.stack(x, dim=0).contiguous()  # type: ignore
    return x

  return transform

def transform_validation(
    resize_size: Tuple[int, int],
    dinov2: bool = True,
    convert_to_rgb: bool = False,
):
  def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

  def identity(x): return x

  resize_and_augment = T.Compose([
      to_rgb if convert_to_rgb else identity,
      T.Resize(resize_size),
      T.ToTensor()
  ])
  normalize = normalize_dino() if dinov2 else identity

  transform_individual_frame = T.Compose([resize_and_augment, normalize])

  def transform(frames: List[Image.Image]):
    x = frames
    x = [transform_individual_frame(f) for f in frames]
    return torch.stack(x)  # type: ignore

  return transform