
import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from models.dino_foresight_pca_vae import DINOForesightPCAVAE
from models.dino_raw_vae import DINORawVAE
from recipe import load_frames
from seed import seed_everything, worker_init_function
from transforms import transform_train


class PrecompDataset(Dataset):
  def __init__(self, shard_path, transform=None):
    super(PrecompDataset, self).__init__()
    self.shard_path = shard_path

    self.image_paths = list(
        sorted(
            [p for p in shard_path.iterdir() if p.name.endswith(".png")]
        )
    )
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    [_, img_index] = img_path.stem.split("img")
    if self.transform:
      return (
          self.transform([img_path]),
          self.shard_path.stem,
          img_index
      )
    return [img_path], self.shard_path.stem, img_index


CONFIG_PATH = os.environ.get(
    "CONFIG_PATH",
    "/users/gabrijel/projects/vgg-wm-vae/configs/"
)
CONFIG_NAME = os.environ.get(
    "CONFIG_NAME",
    "default_movi_a.yaml"
)


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name=CONFIG_NAME
)
def main(cfg: DictConfig):
  # --- Setup ---
  shard_path = Path(cfg.input_dir)
  out_path = shard_path.parent.parent / cfg.name / shard_path.stem
  out_path.mkdir(exist_ok=True, parents=True)
  seed_everything(cfg.training.seed)
  # --- VAE ---
  if "pca" in cfg.name:
    vae = DINOForesightPCAVAE(**cfg.model)
  else:
    vae = DINORawVAE(**cfg.model)
  weights = "model"
  ckpt = torch.load(
      cfg.ckpt_path,
      map_location="cpu",
      weights_only=True
  )
  vae.load_state_dict(ckpt[f"{weights}_state_dict"])
  vae = vae.to(cfg.device)
  vae.eval()

  val_transform = Compose([
      load_frames(),
      transform_train(**cfg.transforms.validation)
  ])
  dataset = PrecompDataset(
      Path(cfg.input_dir),
      val_transform
  )
  dataloader = DataLoader(
      dataset,
      batch_size=cfg.batch_size,
      num_workers=cfg.training.num_workers,
      worker_init_fn=partial(worker_init_function, global_rank=0),
      prefetch_factor=2,
  )

  latents = []
  for (rgb, shards, indices) in dataloader:
    match cfg.objective:
      case "VAE":
        deterministic = False
      case _:
        deterministic = True

    with torch.inference_mode():
      out = vae(
          rgb.to(cfg.device),
          deterministic=deterministic
      )
      (
          latent_mean,
          latent_std
      ) = (
          out.latent_dist.mean,
          out.latent_dist.std
      )
      latents.append(out.latents)

    for (mean, std, shard, index) in zip(
        latent_mean,
        latent_std,
        shards,
        indices
    ):
      np.savez(
          out_path / f"feats-mean-std-{index}.npz",
          **{
              "mean": mean.cpu().numpy(),
              "std": std.cpu().numpy()
          }
      )

  latents = torch.cat(latents, dim=0)
  latent_mean = latents.mean(dim=[0, 1, 2])
  latent_std = latents.std(dim=[0, 1, 2])
  latent_num, *_ = latents.shape

  np.savez(
      shard_path.parent.parent / cfg.name / f"{shard_path.stem}_stats.npz",
      **{
          "mean": latent_mean.detach().cpu().numpy(),
          "std": latent_std.detach().cpu().numpy(),
          "size": latent_num
      }
  )
  print("done")


if __name__ == '__main__':
  main()
