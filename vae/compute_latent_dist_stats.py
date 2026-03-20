import os
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.transforms import Compose

# --- Model ---
from datasets.kubric import KubricDataset
from models.dino_foresight_pca_vae import DINOForesightPCAVAE
from models.dino_raw_vae import DINORawVAE
from seed import seed_everything
from transforms import transform_train

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


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
    """Main entry point for sampling, configured by Hydra."""
    seed_everything(cfg.training.seed)
    device = torch.device("cuda:0")
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

    transform = Compose([
        transform_train(**cfg.transforms.validation)
    ])
    train_dataset = KubricDataset(
        **cfg.data,
        transform=transform,
        split="train"
    )


    rgb_ctx, scene = train_dataset[cfg.dataset_idx]
    rgb_ctx = rgb_ctx[None]  # type: ignore # add batch dim
    rgb_ctx = rgb_ctx.to(device)

    scene_output_dir = (output_dir / scene / f"seed={cfg.training.seed}")
    scene_output_dir.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        b, t, c, h, w = 
        rgb = rgb_ctx.to(device)
        out = model(rgb, deterministic=True)
        sample = einops.rearrange(
            out.sample,
            "(b t) h w c -> b t h w c",
            b=1
        )
        sample = model.postprocess(sample)
        np.save(
            scene_output_dir / f"f_pred.npy",
            (
                sample[0]
                .detach()
                .cpu()
                .numpy()
            )
        )
        symlink_path = scene_output_dir / "f_ctx.npy"
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(scene_output_dir / f"f_pred.npy")


if __name__ == '__main__':
    main()
