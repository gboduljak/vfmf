import os
from functools import partial
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import Compose, ToPILImage

# --- Model ---
from datasets.cityscapes import Cityscapes, CityscapesFrameDataset
from datasets.kubric import KubricDataset
from models.dino_foresight_pca_vae import DINOForesightPCAVAE
from models.dino_raw_vae import DINORawVAE
from recipe import load_frames, sample_frames
from seed import seed_everything
from transforms import denormalize_dino, transform_train

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


config_path = os.environ.get(
    "CONFIG_PATH",
    "/users/gabrijel/projects/vgg-wm-vae/configs/"
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "default_movi_a.yaml"
)


def tensor_to_pil_sequence(rgb: torch.Tensor):
  """
  Converts a normalized DINO RGB tensor (b, t, 3, h, w)
  into a list of denormalized PIL.Image sequences.

  Returns: list[list[PIL.Image]]
           Outer list over batch, inner list over frames.
  """
  if rgb.ndim != 5 or rgb.size(2) != 3:
    raise ValueError(f"Expected shape (b, t, 3, h, w), got {rgb.shape}")

  denorm = denormalize_dino()
  to_pil = ToPILImage()

  rgb = rgb.detach().cpu().clamp(-3, 3)  # avoid extreme values

  pil_sequences = []
  for batch_idx in range(rgb.size(0)):
    frames = []
    for t in range(rgb.size(1)):
      frame = denorm(rgb[batch_idx, t])
      frame = frame.clamp(0, 1)
      frames.append(to_pil(frame))
    pil_sequences.append(frames)
  return pil_sequences


def save_tensor_as_gif(rgb: torch.Tensor, filename: str, fps: int = 10):
  """
  Saves a normalized DINO RGB tensor (b, t, 3, h, w) as a looping GIF.
  Only the first element in batch is used.
  """
  pil_sequences = tensor_to_pil_sequence(rgb)
  frames = pil_sequences[0]  # use first sequence in batch

  duration = int(1000 / fps)  # milliseconds per frame
  frames[0].save(
      filename,
      save_all=True,
      append_images=frames[1:],
      duration=duration,
      loop=0,  # loop forever
  )


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
  """Main entry point for sampling, configured by Hydra."""
  seed_everything(cfg.training.seed)
  device = torch.device("cuda:0" if "device" not in cfg else cfg.device)
  weights = "model"

  output_dir = (
      Path(cfg.output_dir) /
      cfg.name /
      weights
  )
  output_dir.mkdir(exist_ok=True, parents=True)

  if "pca" in cfg.name:
    model = DINOForesightPCAVAE(**cfg.model)
  else:
    model = DINORawVAE(**cfg.model)

  ckpt = torch.load(
      cfg.ckpt_path,
      map_location="cpu",
      weights_only=True
  )
  model.load_state_dict(ckpt[f"{weights}_state_dict"])
  model = model.to(device)

  val_transform = Compose([
      sample_frames(1) if "kubric" not in cfg.name else lambda x:x,
      load_frames(),
      transform_train(**cfg.transforms.validation)
  ])
  if "kubric" in cfg.name:
    DatasetFactory = KubricDataset
  else:
    DatasetFactory = partial(CityscapesFrameDataset, benchmark=True)
    # DatasetFactory = Cityscapes

  val_dataset = DatasetFactory(
      **cfg.data,
      transform=val_transform,
      split="validation"
  )
  rgb_ctx, scene = val_dataset[cfg.dataset_idx]
  rgb_ctx = rgb_ctx[None]  # type: ignore # add batch dim
  rgb_ctx = rgb_ctx.to(device)

  scene_output_dir = (output_dir / scene / f"seed={cfg.training.seed}")
  scene_output_dir.mkdir(exist_ok=True, parents=True)

  with torch.inference_mode():
    rgb = rgb_ctx.to(device)
    save_tensor_as_gif(rgb,  scene_output_dir / "f_ctx.gif", fps=30)

    out = model(rgb, deterministic=True)
    sample = einops.rearrange(
        out.sample,
        "(b t) h w c -> b t h w c",
        b=1
    )
    sample = model.postprocess(sample)
    # if "kubric" not in cfg.name:
    #   b, t, h, w, c = sample.shape
    #   sample_upsampled = F.interpolate(
    #       einops.rearrange(sample, "b t h w c -> (b t) c h w"),
    #       size=(2*h, 2*w),
    #       mode="bicubic",
    #       align_corners=False
    #   )
    #   sample = einops.rearrange(
    #       sample_upsampled,
    #       "(b t) c h w -> b t h w c",
    #       b=b,
    #       t=t
    #   )

    np.save(
        scene_output_dir / f"f_pred.npy",
        (
            sample[0]
            .detach()
            .cpu()
            .numpy()
        )
    )
    np.save(
        scene_output_dir / f"f_ctx.npy",
        (
            model.postprocess(model.preprocess(rgb_ctx))[0]
            .detach()
            .cpu()
            .numpy()
        )
    )
    # symlink_path = scene_output_dir / "f_ctx.npy"
    # if symlink_path.exists():
    #   symlink_path.unlink()
    # symlink_path.symlink_to(scene_output_dir / f"f_pred.npy")


if __name__ == '__main__':
  main()
