from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.dino_foresight_dit import MaskTransformer
from models.vae.dino_raw_vae import DINORawVAE


class AutoregressiveDINOForesightLatent(nn.Module):
  def __init__(
      self,
      sequence_length: int,
      img_size: Tuple[int, int],
      patch_size: int,
      separable_attention: bool,
      separable_window_size: int,
      hidden_dim: int,
      heads: int,
      layers: int,
      dropout: float,
      vae: OmegaConf,
  ):
    super(AutoregressiveDINOForesightLatent, self).__init__()
    shape = (
        sequence_length,
        img_size[0] // (patch_size),
        img_size[1] // (patch_size)
    )
    # self.embedding_dim = self.pca_components.shape[0]
    self.patch_size = patch_size
    self.sequence_length = sequence_length
    # load vae
    self.vae = DINORawVAE(**vae.model)  # type: ignore
    weights = "model"
    ckpt = torch.load(
        vae.ckpt_path,
        map_location="cpu",
        weights_only=True
    )
    self.vae.load_state_dict(ckpt[f"{weights}_state_dict"])
    self.vae.eval()
    for param in self.vae.parameters():
      param.requires_grad = False
    # load vae latent dist stats
    stats = np.load(vae.latent_dist_stats_path)
    self.register_buffer(
        "latent_dist_mean",
        torch.from_numpy(stats["mean"])
    )   # persistent=True by default
    self.register_buffer(
        "latent_dist_std",
        torch.from_numpy(stats["std"])
    )
    self.embedding_dim = self.latent_dist_mean.shape[0]  # type: ignore

    self.transformer = MaskTransformer(
        shape=shape,
        embedding_dim=self.embedding_dim,  # type: ignore
        hidden_dim=hidden_dim,
        depth=layers,
        heads=heads,
        mlp_dim=4 * hidden_dim,
        dropout=dropout,
        separable_attention=separable_attention,
        separable_window_size=separable_window_size,
    )

  @torch.no_grad
  def encode(self, x: torch.Tensor):
    # x: [b, t, c, h, w]
    b, t, *_ = x.shape
    z = self.vae(x, deterministic=False).latents
    z = einops.rearrange(
        z,
        "(b t) h w c -> b t h w c",
        b=b,
        t=t
    )
    mean = einops.repeat(self.latent_dist_mean, "c -> () () () () c")
    std = einops.repeat(self.latent_dist_std, "c -> () () () () c")
    return (z - mean) / std

  @torch.no_grad
  def encode_features(self, f: torch.Tensor, deterministic=False):
    # f: [b, t, h, w, c]
    z = self.vae.encode_features(f, deterministic=deterministic)
    mean = einops.repeat(self.latent_dist_mean, "c -> () () () () c")
    std = einops.repeat(self.latent_dist_std, "c -> () () () () c")
    return (z - mean) / std

  @torch.no_grad
  def decode(self, z):
    # z: [b, t, h, w, c]
    b, t, *_ = z.shape
    mean = einops.repeat(self.latent_dist_mean, "c -> () () () () c")
    std = einops.repeat(self.latent_dist_std, "c -> () () () () c")
    z = z * std + mean
    z = einops.rearrange(z, "b t h w c -> (b t) h w c")
    f_norm = self.vae.vae.decode(z)
    f = self.vae.postprocess(f_norm)
    f = einops.rearrange(f, "(b t) h w c -> b t h w c", b=b, t=t)
    return f

  def forward(self, z_ctx: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor):
    b, t_future, h, w, c = z_t.shape
    z = torch.cat([z_ctx, z_t], dim=1)
    v = self.transformer(z, t)
    v = einops.rearrange(
        v,
        "b (t h w) d -> b t h w d",
        h=h,
        w=w
    )
    v = v[:, -t_future:, ...]
    return v
