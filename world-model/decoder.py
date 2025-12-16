from typing import List, Tuple, TypedDict

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dpt import DPTHead


class DiscreteModality(TypedDict):
  logits: torch.Tensor
  labels: torch.Tensor


class ContinuousModality(TypedDict):
  preds: torch.Tensor


class Modalities(TypedDict):
  fg_bg: DiscreteModality
  segm: DiscreteModality
  depth: DiscreteModality
  surface_normals: ContinuousModality


class ModalityDecoder(nn.Module):
  def __init__(
      self,
      image_size: Tuple[int, int],
      resize_size: Tuple[int, int],
      num_classes: int,
      embed_dim: int,
      dpt_num_features: int,
      dpt_use_bn: bool,
      dpt_out_channels: List[int],
      dpt_layers: List[int],
      dpt_use_clstoken: bool,
      head_ckpt: str,
  ) -> None:
    super().__init__()
    self.dpt_layers = dpt_layers
    self.embed_dim = embed_dim
    self.head = DPTHead(
        nclass=num_classes,
        in_channels=embed_dim,
        features=dpt_num_features,
        use_bn=dpt_use_bn,
        out_channels=dpt_out_channels,
        use_clstoken=dpt_use_clstoken
    )
    head_ckpt = torch.load(head_ckpt, weights_only=False)
    self.head.load_state_dict({
        k.replace("head.", ""): v
        for (k, v) in head_ckpt["state_dict"].items()  # type: ignore
    })
    for param in self.head.scratch.refinenet4.resConfUnit1.parameters():  # type: ignore
      param.requires_grad = False
    self.head.eval()
    self.image_size = image_size
    self.resize_size = resize_size
    self.patch_size = 14
    self.patch_h = image_size[0] // self.patch_size
    self.patch_w = image_size[1] // self.patch_size

  @torch.no_grad
  def forward(self, x):
    # x: [b, t, h, w, d]
    b, t, h, w, d = x.shape
    x = einops.rearrange(
        x,
        "b t h w d -> (b t) (h w) d"
    )
    x = [
        x[..., i * self.embed_dim:(i + 1) * self.embed_dim]
        for i in range(len(self.dpt_layers))
    ]
    x = self.head(x, self.patch_h, self.patch_w)
    x = F.interpolate(
        x,
        size=self.resize_size,
        mode='bicubic',
        align_corners=False
    )
    x = einops.rearrange(
        x,
        "(b t) c h w -> b t c h w",
        b=b
    )
    return x


class ModalitiesDecoder(nn.Module):
  def __init__(
      self,
      fg_bg_ckpt: str,
      segm_ckpt: str,
      depth_ckpt: str,
      normals_ckpt: str,
      image_size: Tuple[int, int],
      resize_size: Tuple[int, int],
      embed_dim: int,
      dpt_num_features: int,
      dpt_use_bn: bool,
      dpt_out_channels: List[int],
      dpt_layers: List[int],
      dpt_use_clstoken: bool,
      num_segm_classes: int
  ):
    super(ModalitiesDecoder, self).__init__()
    # self.fg_bg_decoder = ModalityDecoder(
    #     image_size,
    #     num_classes=2,
    #     embed_dim=embed_dim,
    #     dpt_num_features=dpt_num_features,
    #     dpt_use_bn=dpt_use_bn,
    #     dpt_out_channels=dpt_out_channels,
    #     dpt_use_clstoken=dpt_use_clstoken,
    #     dpt_layers=dpt_layers,
    #     head_ckpt=fg_bg_ckpt
    # )
    self.segm_decoder = ModalityDecoder(
        image_size,
        resize_size,
        num_classes=num_segm_classes,
        embed_dim=embed_dim,
        dpt_num_features=dpt_num_features,
        dpt_use_bn=dpt_use_bn,
        dpt_out_channels=dpt_out_channels,
        dpt_use_clstoken=dpt_use_clstoken,
        dpt_layers=dpt_layers,
        head_ckpt=segm_ckpt
    )
    self.depth_decoder = ModalityDecoder(
        image_size,
        resize_size,
        num_classes=256,
        embed_dim=embed_dim,
        dpt_num_features=dpt_num_features,
        dpt_use_bn=dpt_use_bn,
        dpt_out_channels=dpt_out_channels,
        dpt_use_clstoken=dpt_use_clstoken,
        dpt_layers=dpt_layers,
        head_ckpt=depth_ckpt
    )
    self.normals_decoder = ModalityDecoder(
        image_size,
        resize_size,
        num_classes=3,
        embed_dim=embed_dim,
        dpt_num_features=dpt_num_features,
        dpt_use_bn=dpt_use_bn,
        dpt_out_channels=dpt_out_channels,
        dpt_use_clstoken=dpt_use_clstoken,
        dpt_layers=dpt_layers,
        head_ckpt=normals_ckpt
    )

  def forward(self, f: torch.Tensor) -> Modalities:
    # f: [b, t, h, w, d]
    # fg_bg_logits = self.fg_bg_decoder(f)  # [b, t, c, h, w]
    # fg_bg_preds = torch.argmax(fg_bg_logits, dim=2)  # [b, t, h, w]
    # fg_bg_preds = fg_bg_preds.byte()
    segm_logits = self.segm_decoder(f)  # [b, t, c, h, w]
    segm_preds = torch.argmax(segm_logits, dim=2)  # [b, t, h, w]
    segm_preds = segm_preds.byte()

    depth_logits = self.depth_decoder(f)
    depth_preds = torch.argmax(depth_logits, dim=2)
    depth_preds = depth_preds.byte()

    normals_preds = self.normals_decoder(f)  # [b, t, c, h, w]
    normals_preds = F.normalize(
        normals_preds,
        dim=2,
        p=2
    )  # normals are unit length by convention

    return Modalities(
        fg_bg={},
        segm=DiscreteModality(
            logits=segm_logits.cpu(),
            labels=segm_preds.cpu()
        ),
        depth=DiscreteModality(
            logits=depth_logits.cpu(),
            labels=depth_preds.cpu()
        ),
        surface_normals=ContinuousModality(
            preds=normals_preds.cpu()
        )
    )
