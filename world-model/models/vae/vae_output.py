from typing import NamedTuple

import torch

from models.vae.diagonal_gaussian import DiagonalGaussianDistribution


class VAEOutput(NamedTuple):
  sample: torch.Tensor
  latents: torch.Tensor
  elbo: torch.Tensor
  ll: torch.Tensor
  kl: torch.Tensor
  latent_dist: DiagonalGaussianDistribution
