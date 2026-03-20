import os
from functools import partial
from pathlib import Path
from typing import List

import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_dct import dct_2d
from torchvision.transforms import Compose
from tqdm import tqdm

from datasets.cityscapes import CityscapesFrameDataset
from datasets.ilsvrc import ILSVRC
from datasets.kubric import KubricDataset
from models.dino_foresight_pca_vae import DINOForesightPCAVAE
from models.dino_raw_vae import DINORawVAE
from recipe import load_frames
from seed import seed_everything, worker_init_function
from transforms import transform_train


class IdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
config_path = os.environ.get(
    "CONFIG_PATH",
    "/users/gabrijel/projects/vgg-wm-vae/configs/"
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "default_movi_a.yaml"
)


def channel_wise_2d_dct(x):
    """
    x: (B, H, W, D)
    Returns: (B, H, W, D) DCT applied per (H, W) per channel, independently.
    """
    b, *_ = x.shape

    x = rearrange(x, 'b h w d -> (b d) h w')
    d = dct_2d(x, norm="ortho")
    d = rearrange(d, '(b d) h w -> b h w d', b=b)

    return d

def zigzag_indices(n):
    coords = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            # even: go up-right (start by moving right)
            for i in range(s+1):
                j = s - i
                if i < n and j < n:
                    coords.append((j, i))
        else:
            # odd: go down-left
            for i in range(s+1):
                j = s - i
                if i < n and j < n:
                    coords.append((i, j))
    return coords

def downsample(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize a BHWC tensor to target spatial resolution using bilinear downsampling with einops.

    Args:
        x: Input tensor of shape (B, H, W, C)
        target_h: Target height
        target_w: Target width

    Returns:
        Tensor of shape (B, target_h, target_w, C)
    """
    # Rearrange to BCHW
    x = rearrange(x, 'b h w c -> b c h w')
    # Interpolate
    x = F.interpolate(
        x,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    # Rearrange back to BHWC
    x = rearrange(x, 'b c h w -> b h w c')
    return x

def zigzag_indices_rect(h, w):
  """
  Generates a smoother zigzag traversal over a (h, w) grid.
  Unlike the standard zigzag (which jumps across diagonals),
  this version flows continuously, minimizing distance between consecutive points.

  Returns:
      coords: list of (i, j) indices
  """
  coords = []
  diagonals = []

  # collect diagonals
  for s in range(h + w - 1):
    diag = [(i, s - i)
            for i in range(max(0, s - w + 1), min(h, s + 1)) if 0 <= s - i < w]
    diagonals.append(diag)

  # now traverse them smoothly
  for idx, diag in enumerate(diagonals):
    if idx % 2 == 0:
      # even diagonal: top-down (standard zigzag direction)
      coords.extend(diag)
    else:
      # odd diagonal: reversed, but start from the nearest previous point
      # ensures continuity (no large jumps)
      last = coords[-1] if coords else None
      if last and abs(diag[0][0] - last[0]) + abs(diag[0][1] - last[1]) > abs(diag[-1][0] - last[0]) + abs(diag[-1][1] - last[1]):
        diag = list(reversed(diag))
      coords.extend(diag)

  return coords

def compute_freq_profile(x: torch.Tensor):
    # x: [b, h, w, d]
    b, h, w, d = x.shape
    dct = channel_wise_2d_dct(x)
    if h == w:
        zigzag_coords = torch.tensor(zigzag_indices(h))
    else:
        zigzag_coords = torch.tensor(zigzag_indices_rect(h, w))
    zigzag_index = zigzag_coords[:, 0] * h + zigzag_coords[:, 1]
    # sort dct by zigzag order
    dct_sorted = rearrange(dct, "b h w d -> b (h w) d")
    dct_sorted = dct_sorted[:, zigzag_index, :]
    # compute frequency profile
    dct_profile = torch.abs(dct_sorted / dct_sorted[:, [0], :])
    return dct_profile

class DINOExtractor(nn.Module):
    def __init__(
        self,
        dinov2_variant: str,
        intermediate_layers: List[int],
        patch_size: int,
    ):
        super(DINOExtractor, self).__init__()
        self.dino_v2 = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_' + dinov2_variant,
            pretrained=True
        )
        self.dino_v2.eval()
        for param in self.dino_v2.parameters():
            param.requires_grad = False
        self.d_layers = intermediate_layers
        self.d_num_layers = len(self.d_layers)
        self.patch_size = patch_size

    def extract_features(self, x, reshape=False):
      with torch.no_grad():
        if self.d_layers != [-1]:
          x = self.dino_v2.get_intermediate_layers(
              x,
              n=self.d_layers,
              reshape=reshape
          )
          if self.d_num_layers > 1:
            x = torch.cat(x, dim=-1)
          else:
            x = x[0]
        else:
          x = self.dino_v2.forward_features(x)['x_norm_patchtokens']
      return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W].
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1)  # x.shape [B*T,C,H,W]
        x = self.extract_features(x)  # [B*T,H*W,C]

        x = einops.rearrange(
            x,
            'b (h w) c -> b h w c',
            h=H // self.patch_size,
            w=W // self.patch_size
        )
        x = x.unflatten(dim=0, sizes=(B, T))  # [B,T,H,W,C]
        return x


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
  seed_everything(cfg.training.seed)
  device = torch.device(cfg.device)

  print("--- Configuration ---")
  print(OmegaConf.to_yaml(cfg))
  print("---------------------")

  out_dir = Path(cfg.out_dir)
  analysis_dir = (
    out_dir /
    cfg.name
  )
  analysis_dir.mkdir(exist_ok=True, parents=True)
  
  if cfg.model.name == "identity":
      model = IdentityModel()
  elif "pca" in cfg.name:
      model = DINOForesightPCAVAE(**cfg.model)
  else:
      model = DINORawVAE(**cfg.model)

  # --- Load VAE ---
  if not isinstance(model, IdentityModel):
    print(f"loading {cfg.ckpt_path}...")
    weights = "model"
    ckpt = torch.load(
        cfg.ckpt_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(ckpt[f"{weights}_state_dict"])
    model.to(device)
    model.eval()

  dino = DINOExtractor(
      cfg.model.dinov2_variant,
      cfg.model.intermediate_layers,
      cfg.model.patch_size,
  )
  dino.to(device)
  dino.eval()

  if "kubric" in cfg.data.dataset_root:
    DatasetFactory = KubricDataset
  elif "ILSVRC" in cfg.data.dataset_root:
    DatasetFactory = ILSVRC
  else:
    DatasetFactory = CityscapesFrameDataset

  eval_transform = Compose([
      load_frames(),
      transform_train(**cfg.transforms.validation)
  ])

  if "cityscapes" in cfg.data.dataset_root:
    eval_dataset = DatasetFactory(
        **cfg.data,
        transform=eval_transform,
        split="validation",
        size_limit=15000  # type: ignore
    )
  elif "ILSVRC" in cfg.data.dataset_root:
    eval_dataset = DatasetFactory(
        **cfg.data,
        transform=eval_transform,
        split="val",
    )
  else:
    eval_dataset = DatasetFactory(
        **cfg.data,
        transform=eval_transform,
        split="validation"
    )

  eval_dataloader = DataLoader(
      eval_dataset,
      batch_size=cfg.training.validation_batch_size,
      num_workers=cfg.training.num_workers,
      worker_init_fn=partial(worker_init_function, global_rank=0),
      prefetch_factor=2,
  )

  rgb_profiles = []
  dino_latents_profile = []
  dino_profiles = []
  dino = dino.to(device)
  latents = []

  for batch in tqdm(eval_dataloader):
      rgb, seq = batch

      if isinstance(model, IdentityModel):          
        with torch.inference_mode():
            f = dino(rgb.to(device))
            f = rearrange(f, "b t h w c -> (b t) h w c")
            dino_profiles.append(
                compute_freq_profile(f).cpu()
            )
        rgb = rearrange(rgb, "b t c h w -> (b t) h w c")
        rgb_down = downsample(rgb, *cfg.model.shape)
        rgb_profiles.append(
            compute_freq_profile(rgb_down).cpu()
        )
      else:
        with torch.inference_mode():
          out = model(
            rgb.to(device),
            deterministic=cfg.objective == "AE"
          )
          dino_latents_profile.append(
              compute_freq_profile(out.latents).cpu()
          )
          latents.append(out.latents.cpu())

  if isinstance(model, IdentityModel):
    rgb_profile = torch.cat(rgb_profiles, dim=0)
    dino_profile = torch.cat(dino_profiles, dim=0)
    np.save(
      analysis_dir / "rgb_frequency_profile.npy",
      rgb_profile.detach().cpu().numpy()
    )
    np.save(
      analysis_dir / "dino_frequency_profile.npy",
      dino_profile.detach().cpu().numpy()
    )
  else:          
    dino_latents_profile = torch.cat(dino_latents_profile, dim=0)
    np.save(
      analysis_dir / "frequency_profile.npy",
      dino_latents_profile.detach().cpu().numpy()
    )

if __name__ == '__main__':
  main()
