import os
from copy import deepcopy
from functools import partial
from logging import Logger
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import torch
import torch.distributed as dist
import wandb
from einops import rearrange
from lpips import LPIPS
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torchvision import utils
from torchvision.transforms import Compose, ToPILImage

from datasets.cityscapes import CityscapesFrameDataset
from datasets.ilsvrc import ILSVRC
from datasets.kubric import KubricDataset
from datasets.vspw import VSPWFrameDataset
from ddp import cleanup, setup_ddp
from log import create_logger
from models.dino_raw_vae import DINORawVAE
from models.dino_rgb_decoder import DINORGBDecoder, ImageMetrics, compute_loss
from models.pca_ae import PCAAE
from recipe import load_frames, sample_frame
from scheduler import LinearWarmupScheduler
from seed import seed_everything, worker_init_function
from transforms import denormalize_dino, transform_train

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def plot_dino_recons_side_by_side(recons: torch.Tensor, gt: torch.Tensor, max_samples: int = 8) -> Image.Image:
    assert recons.shape == gt.shape, "Shape mismatch between recons and gt"
    B = min(gt.shape[0], max_samples)
    # denormalize DINO
    denorm = denormalize_dino()
    recons_dn = denorm(recons[:B].cpu()).clamp(0, 1)
    gt_dn = denorm(gt[:B].cpu()).clamp(0, 1)
    # stack pairs along width → (2B, C, H, W)
    pairs = torch.stack([gt_dn, recons_dn], dim=1).flatten(0, 1)
    # make grid: each row = [GT | Recon]
    grid = utils.make_grid(pairs, nrow=2, padding=2)
    # convert to PIL
    to_pil = ToPILImage()
    return to_pil(grid)


def save(
    path: Path,
    model: DINORGBDecoder,
    ema: DINORGBDecoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    epoch: int,
    step: int,
    best_val_psnr: float,
    cfg: DictConfig
):

    if cfg.training.wandb:
        wandb_info = {
            "wandb": {
                "run": {
                    "id": wandb.run.id,
                    "name": wandb.run.name
                }
            }
        }
    else:
        wandb_info = {}

    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_psnr": best_val_psnr,
        **wandb_info
    }, path)


def training_step(
    model: DDP,
    lpips_model: LPIPS,
    batch: List[torch.Tensor],
    device: torch.device,
    dino_vae: Optional[PCAAE] = None,
) -> ImageMetrics:
    rgb, *_ = batch
    rgb = rgb.to(device)
    return compute_loss(
        model(
            rgb,
            partial(autoencode, dino_vae=dino_vae)
        ),
        rgb,
        lpips_model,
    )


@torch.inference_mode()
def evaluation_step(
    model: DINORGBDecoder,
    lpips_model: LPIPS,
    batch: List[torch.Tensor],
    device: torch.device,
    dino_vae: Optional[PCAAE] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb, *_ = batch
    b, t, c, h, w = rgb.shape
    if t != 1:
        rgb = rearrange(
            rgb,
            "b t c h w -> (b t) () c h w"
        )
    rgb = rgb.to(device)
    metrics = compute_loss(
        model(
            rgb,
            partial(autoencode, dino_vae=dino_vae)
        ),
        rgb,
        lpips_model
    )
    return (
        metrics.psnr,
        metrics.ssim,
        metrics.lpips
    )


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for (n_e, p_e), (n_m, p_m) in zip(ema_model.named_parameters(), model.named_parameters()):
        p_e.mul_(decay)
        p_e.add_(p_m.data, alpha=1 - decay)


def evaluate(
    model: DINORGBDecoder,
    lpips_model: LPIPS,
    val_dataloader: DataLoader,
    mixed_precision: bool,
    device: torch.device,
    dino_vae: Optional[PCAAE] = None,
):
    num_examples = 0
    (running_psnr, running_ssim, running_pips) = (0, 0, 0)
    for batch in val_dataloader:
        x, *_ = batch
        b, *_ = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            (psnr, ssim, pips) = evaluation_step(
                model,
                lpips_model,
                batch,
                device,
                dino_vae
            )
        running_psnr += psnr * b
        running_ssim += ssim * b
        running_pips += pips * b
        num_examples += b
    # Calculate the final averages by dividing the running sums by the total
    # number of examples
    return (
        running_psnr / num_examples,
        running_ssim / num_examples,
        running_pips / num_examples
    )

@torch.no_grad
def autoencode(f: torch.Tensor, dino_vae: Optional[PCAAE | DINORawVAE] = None) -> torch.Tensor:
    if dino_vae is None:
        return f
    b, *_ = f.shape
    if isinstance(dino_vae, PCAAE):
        f = rearrange(f, "b t h w c -> (b t) h w c")
        z = dino_vae.encode(f)
        f = dino_vae.decode(z)
        f = rearrange(f, "(b t) h w c -> b t h w c", b=b)
    if isinstance(dino_vae, DINORawVAE):
        z = dino_vae.encode(f, deterministic=False)
        f = dino_vae.decode(z)
    return f

def train(
    model: DDP,
    lpips_model: LPIPS,
    ema: DINORGBDecoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    train_dataloader: DataLoader,
    train_sampler: DistributedSampler,
    val_dataloader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    logger: Logger,
    rank: int,
    initial_epoch: int,
    initial_step: int,
    best_val_psnr: float,
    checkpoint_path: Path,
    samples_path: Path,
    dino_vae: Optional[PCAAE] = None,
):
    model.train()
    ema.eval()  # EMA model should always be in eval mode

    # clean optimizer
    optimizer.zero_grad()
    step = initial_step

    for epoch in range(initial_epoch, cfg.training.epochs):
        train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.training.mixed_precision):
                out = training_step(
                    model,
                    lpips_model,
                    batch,
                    device,
                    dino_vae
                )
                loss = out.loss
                loss = loss / cfg.training.accumulation_steps

            loss.backward()

            if cfg.training.accumulation_steps == 1:
                should_step = True
            else:
                should_step = (step % cfg.training.accumulation_steps == 0)

            if should_step:
                if "clip_grad_norm" in cfg.training:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.clip_grad_norm
                    )
                optimizer.step()
                update_ema(ema, model.module)
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg.training.log_every_steps == 0:
                step_loss = loss * cfg.training.accumulation_steps

                logger.info(
                    f"[epoch={epoch:04d}, step={step:08d}] train_loss={step_loss:.8f} train_l1={out.l1:.8f} train_lpips={out.lpips:.8f}"
                )
                if cfg.training.wandb and rank == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "train_loss": step_loss,
                            "train_l1": out.l1,
                            "train_ssim": out.ssim,
                            "train_lpips": out.lpips,
                            "epoch": epoch,
                            "lr": current_lr
                        },
                        step=step
                    )

            step += 1

        if epoch % cfg.training.evaluate_every_epochs == 0:

            model.eval()
            torch.cuda.empty_cache()
            (psnr, ssim, pips) = evaluate(
                model.module,
                lpips_model,
                val_dataloader,
                cfg.training.mixed_precision,
                device
            )
            torch.cuda.empty_cache()
            model.train()

            dist.all_reduce(psnr, op=dist.ReduceOp.AVG)
            dist.all_reduce(ssim, op=dist.ReduceOp.AVG)
            dist.all_reduce(pips, op=dist.ReduceOp.AVG)

            if rank == 0:
                logger.info(
                    f"[epoch={epoch}] psnr: {psnr.item()}"
                )
                logger.info(
                    f"[epoch={epoch}] ssim: {ssim.item()}")
                logger.info(
                    f"[epoch={epoch}] lpips: {pips.item()}")

                if psnr > best_val_psnr:
                    best_val_psnr = psnr
                    save(
                        checkpoint_path / f"best.pt",
                        model.module,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        best_val_psnr,
                        cfg
                    )
                    logger.info(f"saved checkpoint to {checkpoint_path}")

                if cfg.training.wandb:
                    wandb.log(
                        {"val_psnr": psnr.item()},
                        step=step
                    )
                    wandb.log(
                        {"val_ssim": ssim.item()},
                        step=step
                    )
                    wandb.log(
                        {"val_lpips": pips.item()},
                        step=step
                    )

            dist.barrier()

        if epoch % cfg.training.ckpt_every_epochs == 0:
            if rank == 0:
                save(
                    checkpoint_path / f"current.pt",
                    model.module,
                    ema,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    best_val_psnr,
                    cfg
                )
                logger.info(f"saved checkpoint to {checkpoint_path}")
                
        if epoch % cfg.training.sample_every_epochs == 0:
            for batch in val_dataloader:
                break
            rgb, *_ = batch  # type: ignore
            rgb = rgb[:, [0], ...]
            with torch.inference_mode():
                rgb = rgb.to(device)
                recons = model(rgb)
            if rank == 0:
                plot = plot_dino_recons_side_by_side(
                    recons[:, 0, ...],
                    rgb[:, 0, ...]
                )
                (samples_path /
                 f"rank={rank}").mkdir(exist_ok=True, parents=True)
                plot.save(
                    samples_path / f"rank={rank}" /
                    f"epoch={epoch}-step={step}.png"
                )
                wandb.log({"val_recons": wandb.Image(plot)})

            dist.barrier()


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
    """Main entry point for training, configured by Hydra."""
    # --- Setup DDP and Seeds ---
    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg.training.seed)
    if "vae_config_path" in cfg:
      # Manually load the VAE config
      with open_dict(cfg):
        cfg.vae = OmegaConf.load(cfg.vae_config_path)
        cfg.vae.feature_stats = cfg.model.feature_stats
    if rank == 0:
        print("--- Configuration ---")
        print(OmegaConf.to_yaml(cfg))
        print("---------------------")

    # Set up experiment
    experiment_dir = Path(cfg.experiment_path)
    experiment_dir = experiment_dir / cfg.name
    checkpoint_dir = experiment_dir / "checkpoints"
    samples_dir = experiment_dir / "samples"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    logger = create_logger(experiment_dir, rank)
    logger.info(f"experiment directory created at {experiment_dir}")
    initial_epoch = 0
    initial_step = 0
    # for model selection
    best_val_psnr = 0
    lpips_model = (
        LPIPS(net="vgg")
        .to(device)
        .eval()
    )
    # --- DINO Compression ---
    dino_vae = None
    if "pca" in cfg.name:
      dino_vae = PCAAE(**cfg.vae).to(device).eval()
    if "vae" in cfg.name:
      dino_vae = DINORawVAE(**cfg.vae)
      weights = "model"
      ckpt = torch.load(
          cfg.vae_ckpt_path,
          map_location="cpu",
          weights_only=True
      )
      dino_vae.load_state_dict(ckpt[f"{weights}_state_dict"])
      dino_vae = dino_vae.to(device).eval()
    # --- Model ---
    model = DINORGBDecoder(**cfg.model)
    # Create an EMA of the model for use after training
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    if rank == 0:
        summary(model, depth=3)
    # --- A ---
    if "resume" in cfg:
        ckpt = torch.load(
            cfg.resume.ckpt,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        logger.info(f"restored ckpt: {cfg.resume.ckpt}")
        initial_epoch = ckpt["epoch"] + 1
        initial_step = ckpt["step"] + 1
        best_val_psnr = ckpt["best_val_psnr"]

    if cfg.training.wandb:
        if "resume" in cfg and cfg.resume.wandb:
            if rank == 0:
                wandb.init(
                    project="vgg-wm-vae",
                    config=OmegaConf.to_container(
                        cfg,
                        resolve=True
                    ),  # type: ignore
                    resume="must",
                    id=ckpt["wandb"]["run"]["id"]
                )
        else:
            if rank == 0:
                wandb.init(
                    project="vgg-wm-vae",
                    config=OmegaConf.to_container(
                        cfg,
                        resolve=True
                    ),  # type: ignore
                )
    # Wrap in DDP
    model = model.to(device)
    ema = ema.to(device)
    model = DDP(model)
    # --- Create Dataset and Sampler ---
    train_transform = Compose([
        sample_frame(),
        load_frames(),
        transform_train(**cfg.transforms.train)
    ])
    val_transform = Compose([
        load_frames(),
        transform_train(**cfg.transforms.validation)
    ])
    if "kubric" in cfg.data.dataset_root:
        DatasetFactory = KubricDataset
    elif "vspw" in cfg.data.dataset_root:
        DatasetFactory = VSPWFrameDataset
    elif "ILSVRC" in cfg.data.dataset_root:
        DatasetFactory = ILSVRC
    else:
        DatasetFactory = CityscapesFrameDataset
    train_dataset = DatasetFactory(
        **cfg.data,
        transform=train_transform,
        split="train"
    )
    if "cityscapes" in cfg.data.dataset_root:
        val_dataset = DatasetFactory(
            **cfg.data,
            transform=val_transform,
            split="test",
            size_limit=15000  # type: ignore
        )
    elif "ILSVRC" in cfg.data.dataset_root:
        val_dataset = DatasetFactory(
            **cfg.data,
            transform=val_transform,
            split="test",
        )
    else:
        val_dataset = DatasetFactory(
            **cfg.data,
            transform=val_transform,
            split="validation"
        )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    batch_size = int(
        cfg.training.effective_batch_size /
        (world_size * cfg.training.accumulation_steps)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        worker_init_fn=partial(worker_init_function, global_rank=rank),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.validation_batch_size,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        worker_init_fn=partial(worker_init_function, global_rank=rank),
        prefetch_factor=2,
    )

    logger.info(f"num_steps_per_epoch: {len(train_dataloader)}")
    logger.info(f"local_batch_size: {batch_size}")

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    if "resume" in cfg:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
        logger.info(f"restored optimizer: {cfg.resume.ckpt}")
        if "reset_lr" in cfg.resume:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.resume.reset_lr
            logger.info(
                f"reset lr to {cfg.resume.reset_lr}."
            )
    else:
        # Prepare models for training:
        # Ensure EMA is initialized with synced weights
        update_ema(ema, model.module, decay=0)

    match cfg.training.lr_scheduler.name:
        case "cosine":
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cfg.training.steps
            )
        case "linear_warmup":
            scheduler = LinearWarmupScheduler(
                optimizer=optimizer,
                warmup_steps=cfg.training.lr_scheduler.warmup_steps,
                lr_max=cfg.training.lr
            )
        case _: raise NotImplementedError()

    if "resume" in cfg:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])  # type: ignore
        logger.info(f"restored scheduler: {cfg.resume.ckpt}")

    train(
        model,
        lpips_model,
        ema,
        optimizer,
        scheduler,
        train_dataloader,
        train_sampler,
        val_dataloader,
        cfg,
        device,
        logger,
        rank,
        initial_epoch,
        initial_step,
        best_val_psnr,
        checkpoint_dir,
        samples_dir,
        dino_vae,
    )

    cleanup()


if __name__ == '__main__':
    main()
