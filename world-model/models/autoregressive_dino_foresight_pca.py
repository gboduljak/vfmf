from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn

from models.dino_foresight_dit import MaskTransformer


class AutoregressiveDINOForesightPCA(nn.Module):
  def __init__(
      self,
      dinov2_variant: str,
      pca_ckpt: str,
      intermediate_layers: List[int],
      sequence_length: int,
      img_size: Tuple[int, int],
      patch_size: int,
      separable_attention: bool,
      separable_window_size: int,
      hidden_dim: int,
      heads: int,
      layers: int,
      dropout: float,
      pca_stats: Optional[str] = None,
  ):
    super(AutoregressiveDINOForesightPCA, self).__init__()
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
    pca_dict = torch.load(pca_ckpt, weights_only=False)
    pca = pca_dict['pca_model']
    shape = (
        sequence_length,
        img_size[0] // (patch_size),
        img_size[1] // (patch_size)
    )
    self.pca = True
    self.pca_mean = torch.nn.Parameter(
        torch.tensor(pca.mean_),
        requires_grad=False
    )
    self.pca_components = torch.nn.Parameter(
        torch.tensor(pca.components_),
        requires_grad=False
    )
    self.mean = torch.nn.Parameter(
        torch.tensor(pca_dict['mean']),
        requires_grad=False
    )
    self.std = torch.nn.Parameter(
        torch.tensor(pca_dict['std']),
        requires_grad=False
    )
    if pca_stats is not None:
      stats = np.load(pca_stats)
      self.pca_per_channel_mean = torch.from_numpy(stats['mean'])
      self.pca_per_channel_std = torch.from_numpy(stats['std'])
      self.epsilon = 1e-8
    else:
      self.pca_per_channel_mean = None
      self.pca_per_channel_std = None

    self.embedding_dim = self.pca_components.shape[0]
    self.patch_size = patch_size
    self.sequence_length = sequence_length
    self.transformer = MaskTransformer(
        shape=shape,
        embedding_dim=self.embedding_dim,
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

  def pca_transform(self, x):
    BT, HW, C = x.shape
    x = (x - self.mean) / self.std
    x = x - self.pca_mean
    x_pca = torch.matmul(x, self.pca_components.T)
    if self.pca_per_channel_mean is not None:
      mean = (
          self.pca_per_channel_mean
          .view((1, 1, -1))
          .to(x.device)
      )
      std = (
          self.pca_per_channel_std
          .view((1, 1, -1))
          .to(x.device)
      ) + self.epsilon
      x_pca = (x_pca - mean) / std
    return x_pca

  def pca_inverse_transform(self, x):
    B, T, H, W, C = x.shape
    if self.pca_per_channel_mean is not None:
      mean = (
          self.pca_per_channel_mean
          .view((1, 1, 1, 1, -1))
          .to(x.device)
      )
      std = (
          self.pca_per_channel_std
          .view((1, 1, 1, 1, -1))
          .to(x.device)
      ) + self.epsilon
      x = x * std + mean
    x = torch.matmul(x, self.pca_components) + self.pca_mean
    x = x * self.std + self.mean
    return x

  def preprocess(self, x):
    B, T, C, H, W = x.shape
    # DINOv2 accepts 4 dimensions [B,C,H,W].
    # We use flatten at batch and time dim of x.
    x = x.flatten(end_dim=1)  # x.shape [B*T,C,H,W]
    x = self.extract_features(x)  # [B*T,H*W,C]
    if self.pca:
      x = self.pca_transform(x)
    x = einops.rearrange(
        x,
        'b (h w) c -> b h w c',
        h=H // self.patch_size,
        w=W // self.patch_size
    )
    x = x.unflatten(dim=0, sizes=(B, T))  # [B,T,H,W,C]
    return x

  def postprocess(self, x):
    if self.pca:
      x = self.pca_inverse_transform(x)
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
