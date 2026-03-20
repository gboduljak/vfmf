

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diagonal_gaussian import DiagonalGaussianDistribution
from models.dino_foresight_vae import VAEOutput


class MlpVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        latent_channels: int,
        shape: List[int],
        beta: float
    ) -> None:
        super().__init__()
        # encoder
        self.in_proj = nn.Linear(input_dim, mlp_dim)
        self.in_proj.weight.data.normal_(std=0.02)
        self.encoder_backbone = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
        )
        self.to_latents_norm = nn.LayerNorm(mlp_dim)
        self.to_latents = nn.Linear(mlp_dim, 2*latent_channels)
        self.to_latents.weight.data.normal_(std=0.02)
        # decoder
        self.from_latents = nn.Linear(latent_channels, mlp_dim)
        self.from_latents.weight.data.normal_(std=0.02)
        self.decoder_backbone = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(mlp_dim, input_dim)
        self.out_proj.weight.data.normal_(std=0.02)
        self.shape = shape
        self.beta = beta

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = x
        h = self.in_proj(h)
        h = self.encoder_backbone(h)
        h = self.to_latents_norm(h)
        return DiagonalGaussianDistribution(self.to_latents(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        h = self.from_latents(h)
        h = self.decoder_backbone(h)
        h = self.out_proj(h)
        return h

    def forward(self, x, deterministic=False):
        q_z_given_x = self.encode(x)
        if deterministic:
            z = q_z_given_x.mode()
        else:
            z = q_z_given_x.sample()

        x_hat = self.decode(z)
        # mathematically, it is correct to use sum instead of mean for both kl and ll
        kl = q_z_given_x.kl()  # [b, ]
        ll = -F.mse_loss(
            x_hat,
            x,
            reduction="none"
        ).sum(dim=[1, 2, 3])  # [b, ]

        if deterministic:
            elbo = ll
        else:
            elbo = ll - self.beta*kl

        return VAEOutput(
            sample=x_hat,
            latents=z,
            elbo=elbo,
            ll=-F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3]),
            kl=q_z_given_x.kl(),
            latent_dist=q_z_given_x
        )
