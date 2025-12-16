
from typing import List, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.vae.attention_masked import AddBroadcastPosEmbed, Transformer
from models.vae.diagonal_gaussian import DiagonalGaussianDistribution


class VAEOutput(NamedTuple):
  sample: torch.Tensor
  latents: torch.Tensor
  elbo: torch.Tensor
  ll: torch.Tensor
  kl: torch.Tensor
  latent_dist: DiagonalGaussianDistribution


class DINOForesightVAE(nn.Module):
  def __init__(
      self,
      num_encoder_layers: int,
      num_decoder_layers: int,
      heads: int,
      input_dim: int,
      hidden_dim: int,
      mlp_dim: int,
      latent_channels: int,
      dropout: float,
      use_qk_norm: bool,
      shape: List[int],
      beta: float,
      abs_pos_enc: bool,
      num_registers: int
  ) -> None:
    super().__init__()
    # encoder
    self.in_proj = nn.Linear(input_dim, hidden_dim)
    self.in_proj.weight.data.normal_(std=0.02)
    if abs_pos_enc:
      self.encoder_pos_enc = AddBroadcastPosEmbed(
          shape=shape,
          embd_dim=hidden_dim
      )
      self.decoder_pos_enc = AddBroadcastPosEmbed(
          shape=shape,
          embd_dim=hidden_dim
      )
    self.encoder_backbone = Transformer(
        dim=hidden_dim,
        depth=num_encoder_layers,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        use_qk_norm=use_qk_norm,
        num_registers=num_registers
    )
    self.to_latents_norm = nn.LayerNorm(hidden_dim)
    self.to_latents = nn.Linear(hidden_dim, 2 * latent_channels)
    self.to_latents.weight.data.normal_(std=0.02)
    # decoder
    self.from_latents = nn.Linear(latent_channels, hidden_dim)
    self.from_latents.weight.data.normal_(std=0.02)
    self.decoder_backbone = Transformer(
        dim=hidden_dim,
        depth=num_decoder_layers,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        use_qk_norm=use_qk_norm,
        num_registers=num_registers
    )
    self.out_proj = nn.Linear(hidden_dim, input_dim)
    self.out_proj.weight.data.normal_(std=0.02)
    self.shape = shape
    self.beta = beta

  def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
    [_, w] = self.shape
    h = self.in_proj(x)
    if hasattr(self, "encoder_pos_enc"):
      h = self.encoder_pos_enc(h)
    h = rearrange(h, "b h w c -> b (h w) c")
    h = self.encoder_backbone(h)
    h = self.to_latents_norm(h)
    h = rearrange(h, "b (h w) c -> b h w c", w=w)
    return DiagonalGaussianDistribution(self.to_latents(h))

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    [_, w] = self.shape
    h = self.from_latents(z)
    if hasattr(self, "decoder_pos_enc"):
      h = self.decoder_pos_enc(h)
    h = rearrange(h, "b h w c -> b (h w) c")
    h = self.decoder_backbone(h)
    h = self.out_proj(h)
    x = rearrange(h, "b (h w) c -> b h w c", w=w)
    return x

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
