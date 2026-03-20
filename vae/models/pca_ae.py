from typing import Tuple

import einops
import torch
import torch.nn as nn


class PCAAE(nn.Module):
    def __init__(
        self,
        pca_ckpt: str,
        pca_rank: int,
        shape: Tuple[int, int] = (16, 16)
    ):
        super(PCAAE, self).__init__()
        pca_stats = torch.load(pca_ckpt, weights_only=False)
        pca_model = pca_stats["pca_model"]
        self.shape = shape
        self.pca_rank = pca_rank
        self.register_buffer(
            "pca_components",
            torch.tensor(
                pca_model.components_[:pca_rank, :],
                dtype=torch.float32
            )
        )
        self.register_buffer(
            "pca_mean",
            torch.tensor(pca_model.mean_, dtype=torch.float32)
        )
        self.register_buffer(
            "mean_dino",
            torch.tensor(pca_stats["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "std_dino",
            torch.tensor(pca_stats["std"], dtype=torch.float32)
        )

    @torch.no_grad
    def encode(self, f):
        z = f
        z = (z - self.mean_dino) / self.std_dino
        z = z - self.pca_mean
        z_pca = z @ self.pca_components.T
        [h, w] = self.shape
        z_pca = z_pca.reshape(-1, h, w, self.pca_rank)
        return z_pca

    @torch.no_grad
    def decode(self, z_pca):
        # z_pca: (b, h, w, c)
        b, h, w, c = z_pca.shape
        z = einops.rearrange(z_pca, "b h w c -> b (h w) c")
        f = (
            torch.matmul(z, self.pca_components) +  # type: ignore
            self.pca_mean  # type: ignore
        )
        f = f * self.std_dino + self.mean_dino  # type: ignore
        f = einops.rearrange(f, "b (h w) c -> b h w c", h=h, w=w)
        return f
