from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn

from models.dino_foresight_dit import MaskTransformer


class AutoregressiveDINOForesight(nn.Module):
  def __init__(
      self,
      dinov2_variant: str,
      intermediate_layers: List[int],
      sequence_length: int,
      img_size: Tuple[int, int],
      patch_size: int,
      separable_attention: bool,
      separable_window_size: int,
      input_dim: int,
      hidden_dim: int,
      heads: int,
      layers: int,
      dropout: float,
      feature_stats: Optional[str] = None,
  ):
    super(AutoregressiveDINOForesight, self).__init__()
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

    shape = (
        sequence_length,
        img_size[0] // (patch_size),
        img_size[1] // (patch_size)
    )

    if feature_stats is not None:
      stats = np.load(feature_stats)
      self.feature_per_channel_mean = torch.from_numpy(stats['mean'])
      self.feature_per_channel_std = torch.from_numpy(stats['std'])
      self.epsilon = 1e-8
    else:
      self.feature_per_channel_mean = None
      self.feature_per_channel_std = None

    self.patch_size = patch_size
    self.sequence_length = sequence_length
    self.transformer = MaskTransformer(
        shape=shape,
        embedding_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=layers,
        heads=heads,
        mlp_dim=4 * hidden_dim,
        dropout=dropout,
        separable_attention=separable_attention,
        separable_window_size=separable_window_size,
    )

  def extract_features(self, x, reshape=False):
    with torch.no_grad():
      x = self.dino_v2.get_intermediate_layers(
          x,
          n=self.d_layers,
          reshape=reshape
      )
      if self.d_num_layers > 1:
        x = torch.cat(x, dim=-1)
      else:
        x = x[0]
    return x

  def feature_transform(self, x):
    mean = (
        self.feature_per_channel_mean
        .view((1, 1, -1))
        .to(x.device)
    )
    std = (
        self.feature_per_channel_std
        .view((1, 1, -1))
        .to(x.device)
    ) + self.epsilon
    return (x - mean) / std

  def feature_inverse_transform(self, x):
    mean = (
        self.feature_per_channel_mean
        .view((1, 1, -1))
        .to(x.device)
    )
    std = (
        self.feature_per_channel_std
        .view((1, 1, -1))
        .to(x.device)
    ) + self.epsilon
    x = x * std + mean
    return x

  def preprocess(self, x):
    B, T, C, H, W = x.shape
    # DINOv2 accepts 4 dimensions [B,C,H,W].
    # We use flatten at batch and time dim of x.
    x = x.flatten(end_dim=1)  # x.shape [B*T,C,H,W]
    x = self.extract_features(x)  # [B*T,H*W,C]
    if self.feature_per_channel_mean is not None:
      x = self.feature_transform(x)
    x = einops.rearrange(
        x,
        'b (h w) c -> b h w c',
        h=H // self.patch_size,
        w=W // self.patch_size
    )
    x = x.unflatten(dim=0, sizes=(B, T))  # [B,T,H,W,C]
    return x

  def postprocess(self, x):
    if self.feature_per_channel_mean is not None:
      x = self.feature_inverse_transform(x)
    return x

  def forward(self, x_ctx: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
    b, t_future, h, w, c = x_t.shape
    x = torch.cat([x_ctx, x_t], dim=1)
    v = self.transformer(x, t)
    v = einops.rearrange(
        v,
        "b (t h w) d -> b t h w d",
        h=h,
        w=w
    )
    v = v[:, -t_future:, ...]
    return v
