import os
from pathlib import Path
from typing import List, Literal, Optional

import einops
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint
from torchinfo import summary
from torchvision.transforms import Compose, ToPILImage

from datasets.cityscapes import CityscapesBenchmarkDataset
from datasets.kubric import KubricMultiDatasetFramesOnly
from models.autoregressive_dino_foresight import AutoregressiveDINOForesight
from models.autoregressive_dino_foresight_latent import \
  AutoregressiveDINOForesightLatent
from models.autoregressive_dino_foresight_pca import \
  AutoregressiveDINOForesightPCA
from recipe import load_frames
from seed import seed_everything
from transforms import denormalize_dino, transform_train


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


@torch.no_grad
def sample(
    velocity: AutoregressiveDINOForesightPCA | AutoregressiveDINOForesightLatent,
    f_ctx: torch.Tensor,
    steps=100,
    method="euler",
    rng: Optional[torch.Generator] = None
):
  # f_ctx: [b, t - 1, h, w, d]
  b, _, h, w, d = f_ctx.shape
  f0 = torch.randn(
      (b, 1, h, w, d),
      device=f_ctx.device,
      generator=rng
  )
  t = torch.linspace(0., 1., steps, device=f_ctx.device)
  # Define ODE

  @torch.no_grad
  def v(t, f_t):
    # predict velocity
    return velocity.forward(
        f_ctx,
        f_t,
        t.expand(b)
    )
  # Integrate ODE
  f_traj = odeint(v, f0, t, method=method)
  return f_traj[-1], f_traj


@torch.no_grad
def sample_autoregressive(
    velocity: AutoregressiveDINOForesightPCA | AutoregressiveDINOForesightLatent,
    f_ctx: torch.Tensor,
    num_rollouts: int,
    max_context_length: int = 4,
    steps=100,
    method="euler",
    rng: Optional[torch.Generator] = None
):
  # f_ctx: [b, t - 1, h, w, d]
  f_preds = []
  print(f"[sampling steps]: {steps}")
  for _ in range(num_rollouts):
    _, context_len, *_ = f_ctx.shape
    print(f"[context length]: {context_len}")
    if isinstance(velocity, AutoregressiveDINOForesightLatent):
      z_ctx = velocity.encode_features(f_ctx)
      z_pred, _ = sample(velocity, z_ctx, steps, method, rng)
      f_pred = velocity.decode(z_pred)
    else:
      f_pred, _ = sample(velocity, f_ctx, steps, method, rng)
    if context_len < max_context_length:
      f_ctx = torch.cat(
          [f_ctx, f_pred],  # type: ignore
          dim=1
      )  # type: ignore
    else:
      f_ctx = torch.cat(
          [f_ctx[:, 1:, ...], f_pred],  # type: ignore
          dim=1
      )  # type: ignore
    f_preds.append(f_pred)

  f_preds = torch.stack(f_preds)
  f_preds = einops.rearrange(
      f_preds,
      "k b 1 h w d -> b k h w d"
  )
  return f_preds


def sample_benchmark_dino_foresight(
    sequence_length: int,
    horizon: Literal["short", "medium", "long"]
):

  match horizon:
    case "short":
      sequence_length = 5
      num_frames_skip = 2
      step = num_frames_skip + 1
      start_idx = 20 - step * sequence_length + num_frames_skip
      gt_idx = 19
    case "medium":
      sequence_length = 5
      num_frames_skip = 2
      step = num_frames_skip + 1
      gt_idx = 19
      start_idx = (
          (gt_idx + 1) - step * sequence_length + num_frames_skip - 6
      )
    case "long":
      sequence_length = 5
      num_frames_skip = 1
      step = num_frames_skip + 1
      gt_idx = 19
      start_idx = (
          (gt_idx + 1) - step * sequence_length + num_frames_skip - 6
      )

  print(
      np.arange(0, 24)[
          start_idx: start_idx + step * sequence_length: step
      ][:-1]
  )

  def sample(frames_path: List[Path]) -> List[Path]:
    return frames_path[
        start_idx: start_idx + step * sequence_length: step
    ][:-1]

  return sample


def sample_benchmark_uncertain(
    stride: int = 2,
    frame_start_idx: int = 1,
    context_length: int = 1
):
  def sample(frames_path: List[Path]) -> List[Path]:
    # rollout is [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    frames = frames_path
    frames = frames[frame_start_idx:]
    frames = frames[::stride]
    frames = frames[:context_length]
    return frames

  return sample


config_path = os.environ.get(
    "CONFIG_PATH",
    "/users/gabrijel/projects/vgg-wm/configs/kubric/"
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "dino_foresight_cfm.yaml"
)


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
  """Main entry point for training, configured by Hydra."""

  example_idx = cfg.example_idx
  horizon = cfg.horizon
  weights = cfg.weights_type
  device = cfg.device
  seeds = cfg.sampling.seeds
  ckpt = Path(cfg.ckpt_path).stem

  sampling_method = "euler"
  sampling_steps = cfg.sampling.get("steps", 10)
  sampling_summary = f"{sampling_method}({sampling_steps})"

  output_dir = (
      Path(cfg.output_dir) /
      cfg.split /
      cfg.name /
      ckpt /
      weights /
      sampling_summary /
      horizon
  )
  if horizon == "uncertain":
    output_dir = (
        output_dir /
        f"initial_context_length={cfg.initial_context_length}"
    )
  output_dir.mkdir(exist_ok=True, parents=True)

  print("--- Configuration ---")
  print(OmegaConf.to_yaml(cfg))
  print("---------------------")

  print("--- Model ---")
  if "pca" in cfg.name:
    model = AutoregressiveDINOForesightPCA(**cfg.model)
  elif "vae" in cfg:
    model = AutoregressiveDINOForesightLatent(**cfg.model, vae=cfg.vae)
  else:
    model = AutoregressiveDINOForesight(**cfg.model)
  summary(model)
  print("---------------------")
  print("--- Model ---")
  ckpt = torch.load(
      cfg.ckpt_path,
      map_location="cpu",
      weights_only=False
  )
  model.load_state_dict(ckpt[f"{weights}_state_dict"])
  model = model.to(device)
  print("---------------------")
  # as argument receive output dir, scene_idx
  match cfg.benchmark:
    case "dino_foresight":
      val_transform = Compose([
          sample_benchmark_dino_foresight(
              sequence_length=cfg.model.sequence_length,
              horizon=horizon
          ),
          load_frames(),
          transform_train(**cfg.transforms.validation)
      ])
    case "dino_foresight_uncertain":
      val_transform = Compose([
          sample_benchmark_uncertain(
              stride=2,
              frame_start_idx=(
                  1 if "cityscapes" in cfg.data.dataset_root else 0
              ),
              context_length=cfg.initial_context_length
          ),
          load_frames(convert_to_rgb="kubric" in cfg.data.dataset_root),
          transform_train(**cfg.transforms.validation)
      ])
    case _:
      raise NotImplementedError()

  if "cityscapes" in cfg.data.dataset_root:
    val_dataset = CityscapesBenchmarkDataset(
        **cfg.data,
        transform=val_transform,
        split=cfg.split,
    )
  else:
    val_dataset = KubricMultiDatasetFramesOnly(
        cfg.data.dataset_root,
        cfg.initial_context_length,
        val_transform,
        max_size=128
    )
  print(example_idx)
  rgb_ctx, scene = val_dataset[example_idx]
  rgb_ctx = rgb_ctx[None]  # add batch dim
  rgb_ctx = rgb_ctx.to(device)

  f_preds: List[torch.Tensor] = []

  if isinstance(model, AutoregressiveDINOForesightLatent):
    f_ctx: torch.Tensor = model.vae.postprocess(model.vae.preprocess(rgb_ctx))
  else:
    print("[preprocessing ctx...]")
    f_ctx: torch.Tensor = model.preprocess(rgb_ctx)

  for seed in seeds:

    seed_everything(seed)
    scene_output_dir = (output_dir / scene / f"seed={seed}")
    scene_output_dir.mkdir(exist_ok=True, parents=True)
    save_tensor_as_gif(
        rgb_ctx,
        scene_output_dir / "f_ctx.gif",
        fps=17 if "cityscapes" in cfg.data.dataset_root else 12
    )

    match cfg.benchmark:
      case "dino_foresight":
        match horizon:
          case "long":
            num_rollouts = 8
          case "medium":
            num_rollouts = 3
          case "short":
            num_rollouts = 1
      case "dino_foresight_uncertain":
        if "cityscapes" in cfg.data.dataset_root:
          num_rollouts = 9 - (cfg.initial_context_length - 1)
        else:
          num_rollouts = 11 - (cfg.initial_context_length - 1)
    max_context_length = 4
    f_pred = sample_autoregressive(
        model,
        f_ctx,
        num_rollouts,  # type: ignore
        max_context_length,
        sampling_steps,
        sampling_method,
        rng=torch.Generator(device).manual_seed(seed)
    )
    if (
        isinstance(model, AutoregressiveDINOForesight) or
        isinstance(model, AutoregressiveDINOForesightPCA)
    ):
      print("[postprocessing pred...]")
      f_pred = model.postprocess(f_pred)
      np.save(
          scene_output_dir / f"f_ctx.npy",
          model.postprocess(f_ctx.clone())[0].cpu().numpy()  # type: ignore
      )
    else:
      np.save(
          scene_output_dir / f"f_ctx.npy",
          f_ctx.clone()[0].cpu().numpy()
      )  # type: ignore

    np.save(
        scene_output_dir / f"f_pred.npy",
        f_pred[0].cpu().numpy()
    )
    f_preds.append(f_pred)

  f_preds_cat = torch.cat(f_preds, dim=0)
  for k in [1, 2, 4, 8, 16, 32, 64]:
    if k <= f_preds_cat.shape[0]:  # only if enough elements exist
      mean_k = f_preds_cat[:k].mean(dim=0)
      mean_k_dir = Path(output_dir) / scene / f"mean_{k:02d}"
      mean_k_dir.mkdir(exist_ok=True, parents=True)
      np.save(mean_k_dir / "f_pred.npy", mean_k.cpu().numpy())
      save_tensor_as_gif(
          rgb_ctx,
          mean_k_dir / "f_ctx.gif",
          fps=17 if "cityscapes" in cfg.data.dataset_root else 12
      )
      if (
          isinstance(model, AutoregressiveDINOForesight) or
          isinstance(model, AutoregressiveDINOForesightPCA)
      ):
        np.save(
            mean_k_dir / f"f_ctx.npy",
            model.postprocess(f_ctx.clone())[0].cpu().numpy()  # type: ignore
        )
      else:
        np.save(
            mean_k_dir / f"f_ctx.npy",
            f_ctx.clone()[0].cpu().numpy()
        )  # type: ignore


if __name__ == "__main__":
  main()
