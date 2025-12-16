

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.vae.convnext import ConvNeXtIsotropic
from models.vae.diagonal_gaussian import DiagonalGaussianDistribution
from models.vae.vae_output import VAEOutput


class ConvNeXtIsotropicVAE(nn.Module):
  def __init__(
      self,
      input_dim: int,
      depth: int,
      dim: int,
      drop_path_rate: float,
      layer_scale_init_value: int,
      latent_channels: int,
      shape: List[int],
      beta: float,
      **kwargs
  ) -> None:
    super().__init__()
    # encoder
    self.encoder_backbone = ConvNeXtIsotropic(
        in_chans=input_dim,
        depth=depth,
        dim=dim,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    self.to_latents_norm = nn.LayerNorm(dim)
    self.to_latents = nn.Linear(dim, 2 * latent_channels)
    self.to_latents.weight.data.normal_(std=0.02)
    # decoder
    self.from_latents = nn.Linear(latent_channels, dim)
    self.from_latents.weight.data.normal_(std=0.02)
    self.decoder_backbone = ConvNeXtIsotropic(
        in_chans=dim,
        depth=depth,
        dim=dim,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    self.out_proj = nn.Linear(dim, input_dim)
    self.out_proj.weight.data.normal_(std=0.02)
    self.shape = shape
    self.beta = beta

  def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
    # x: [b, h, w, c]
    h = rearrange(x, "b h w c -> b c h w").contiguous()
    h = self.encoder_backbone(h)
    h = rearrange(h, "b c h w -> b h w c").contiguous()
    h = self.to_latents_norm(h)
    return DiagonalGaussianDistribution(self.to_latents(h))

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    h = z
    h = self.from_latents(h)
    h = rearrange(h, "b h w c -> b c h w").contiguous()
    h = self.decoder_backbone(h)
    h = rearrange(h, "b c h w -> b h w c").contiguous()
    h = self.out_proj(h)
    return h

  def forward(self, x, deterministic=False):
    q_z_given_x = self.encode(x)
    if deterministic:
      z = q_z_given_x.mode()
    else:
      z = q_z_given_x.sample()

    x_hat = self.decode(z)
    # mathematically, it is correct to use sum instead of mean for both kl and
    # ll
    kl = q_z_given_x.kl()  # [b, ]
    ll = -F.mse_loss(
        x_hat,
        x,
        reduction="none"
    ).sum(dim=[1, 2, 3])  # [b, ]

    if deterministic:
      elbo = ll
    else:
      elbo = ll - self.beta * kl

    return VAEOutput(
        sample=x_hat,
        latents=z,
        elbo=elbo,
        ll=-F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3]),
        kl=q_z_given_x.kl(),
        latent_dist=q_z_given_x
    )
