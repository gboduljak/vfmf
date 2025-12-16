from typing import Callable, List, NamedTuple, Optional, Tuple

import einops
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import \
  structural_similarity_index_measure as ssim

from models.dpt_head import DPTHead
from models.rgb.attention_masked import AddBroadcastPosEmbed, Transformer


class ImageMetrics(NamedTuple):
    loss: torch.Tensor
    l1: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor
    lpips: torch.Tensor


def compute_loss(
    recons: torch.Tensor,
    gt: torch.Tensor,
    lpips_model: lpips.LPIPS
) -> ImageMetrics:
    def _normalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + 1.0) / 2.0
    # recons: [b, t, c, h, w]
    # gt: [b, t, c, h, w]
    b, t, c, h, w = recons.shape
    assert recons.shape == gt.shape
    assert t == 1
    recons = recons[:, 0, ...]
    gt = gt[:, 0, ...]
    # differentiable metrics
    loss_l1 = F.l1_loss(recons, gt)
    loss_lpips = lpips_model(recons, gt).mean()
    # --- PSNR & SSIM ---
    # We normalize our [-1, 1] tensors to [0, 1].
    recons_01 = _normalize_to_01(recons)
    gt_01 = _normalize_to_01(gt)
    with torch.no_grad():
        metric_psnr = psnr(recons_01, gt_01, data_range=1.0)
        metric_ssim = ssim(recons_01, gt_01, data_range=1.0)
    return ImageMetrics(
        loss=loss_l1 + loss_lpips,
        l1=loss_l1,
        psnr=metric_psnr,
        ssim=metric_ssim,  # type: ignore
        lpips=loss_lpips
    )


class DINORGBDecoder(nn.Module):
    def __init__(
        self,
        dinov2_variant: str,
        dino_intermediate_layers: List[int],
        decoder_intermediate_layers: List[int],
        resolution: Tuple[int, int],
        patch_size: int,
        num_layers: int,
        heads: int,
        input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        use_qk_norm: bool,
        shape: List[int],
        num_registers: int,
        feature_stats: Optional[str] = None,
        **kwargs
    ):
        super(DINORGBDecoder, self).__init__()
        self.dino_v2 = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_' + dinov2_variant,
            pretrained=True
        )
        if dino_intermediate_layers == [-1]:
            del self.dino_v2.head
            self.dino_v2.head = torch.nn.Identity()
        self.dino_v2.eval()
        for param in self.dino_v2.parameters():
            param.requires_grad = False
        self.d_layers = dino_intermediate_layers
        self.d_num_layers = len(self.d_layers)
        [h, w] = resolution
        self.height = h
        self.width = w
        if feature_stats is not None:
            stats = np.load(feature_stats)
            self.feature_per_channel_mean = torch.from_numpy(stats['mean'])
            self.feature_per_channel_std = torch.from_numpy(stats['std'])
            self.epsilon = 1e-8
        else:
            self.feature_per_channel_mean = None
            self.feature_per_channel_std = None

        self.patch_size = patch_size
        # encoder
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_proj.weight.data.normal_(std=0.02)
        self.decoder_pos_enc = AddBroadcastPosEmbed(
            shape=shape,
            embd_dim=hidden_dim
        )
        self.decoder_backbone = Transformer(
            dim=hidden_dim,
            depth=num_layers,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            num_registers=num_registers
        )
        self.dpt = DPTHead(
            dim_in=hidden_dim,
            output_dim=3 + 1,
            activation="linear",
            intermediate_layer_idx=decoder_intermediate_layers,
        )

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

    def decode(self, f: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # f: [b, t, h, w, c]
        # !!! Assumes that f is raw features. By default it should normalize for decoder.
        b, t, *_ = f.shape
        if normalize:
            if self.feature_per_channel_mean is not None:
                f = self.feature_transform(f)
        h = self.in_proj(f)
        # h: [b, t, h, w, d]
        if hasattr(self, "decoder_pos_enc"):
            h = einops.rearrange(h, "b t h w d -> (b t) h w d")
            h = self.decoder_pos_enc(h)
            h = einops.rearrange(h, "(b t) h w d -> b t h w d", b=b)
        h = einops.rearrange(h, "b t h w c -> (b t) (h w) c")
        hs = self.decoder_backbone(h, return_intermediate_layers=True)
        hs = [
            einops.repeat(h, "b n d -> b t n d", t=t)
            for h in hs
        ]
        x_hat, *_ = self.dpt.forward(
            aggregated_tokens_list=hs,
            images=torch.zeros(
                (b, t, 3, self.height, self.width),
                device=f.device
            ),
            patch_start_idx=self.decoder_backbone.num_registers
        )
        return einops.rearrange(
            x_hat,
            "b t h w c -> b t c h w"
        )

    def forward(self, x: torch.Tensor, autoencode: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> torch.Tensor:
        # x: [b, t, c, h, w]
        b, t, *_ = x.shape
        assert t == 1
        # --- Extract DINOv2 features
        b, t, c, h, w = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W].
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1)  # x.shape [B*T,C,H,W]
        f = self.extract_features(x)  # [B*T,H*W,C]
        # --- Push through autoencoder
        f = einops.rearrange(
          f,
          "(b t) (h w) c -> b t h w c",
          b=b,
          h=h // self.patch_size,
          w=w // self.patch_size
        )
        f = autoencode(f)
        f = einops.rearrange(
            f,
            'b t h w c  -> (b t) h w c',
            h=h // self.patch_size,
            w=w // self.patch_size
        )
        f = f.unflatten(dim=0, sizes=(b, t))  # [b,t,h,w,c]
        # f: [b, t, h, w, d]
        return self.decode(f, normalize=True)