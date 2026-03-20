import os
from copy import deepcopy
from functools import partial
from logging import Logger
from pathlib import Path
from typing import List

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torchvision.transforms import Compose

from datasets.cityscapes import CityscapesFrameDataset
from datasets.ilsvrc import ILSVRC
from datasets.kubric import KubricDataset
from datasets.vspw import VSPWFrameDataset
from ddp import cleanup, setup_ddp
from log import create_logger
from models.dino_foresight_pca_vae import DINOForesightPCAVAE
from models.dino_raw_vae import DINORawVAE
from recipe import load_frames, sample_frame
from scheduler import LinearWarmupScheduler
from seed import seed_everything, worker_init_function
from transforms import transform_train

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def save(
    path: Path,
    model: DINOForesightPCAVAE,
    ema: DINOForesightPCAVAE,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    epoch: int,
    step: int,
    best_val_l2: float,
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
        "best_val_l2": best_val_l2,
        **wandb_info
    }, path)


def training_step(
    model: DDP,
    deterministic: bool,
    batch: List[torch.Tensor],
    device: torch.device
):
    rgb, *_ = batch
    rgb = rgb.to(device)
    return model(rgb, deterministic)


@torch.inference_mode()
def evaluation_step(
    model: DINOForesightPCAVAE | DINORawVAE,
    deterministic: bool,
    batch: List[torch.Tensor],
    device: torch.device
):
    rgb, *_ = batch
    rgb = rgb.to(device)
    out = model(rgb, deterministic)
    recons = out.sample
    _, h, w, d = recons.shape
    l2 = (-out.ll / (h * w * d)).mean()
    nll = -out.ll.mean()
    kl = out.kl.mean()
    return (l2, nll, kl)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for (n_e, p_e), (n_m, p_m) in zip(ema_model.named_parameters(), model.named_parameters()):
        p_e.mul_(decay)
        p_e.add_(p_m.data, alpha=1 - decay)


def evaluate(
    model: DINOForesightPCAVAE,
    val_dataloader: DataLoader,
    deterministic: bool,
    mixed_precision: bool,
    device: torch.device
):
    num_examples = 0
    (running_l2, running_nll, running_kl) = (0, 0, 0)
    for batch in val_dataloader:
        x, *_ = batch
        b, *_ = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            (l2, nll, kl) = evaluation_step(
                model,
                deterministic,
                batch,
                device
            )
        running_l2 += l2 * b
        running_nll += nll * b
        running_kl += kl * b
        num_examples += b
    # Calculate the final averages by dividing the running sums by the total
    # number of examples
    l2 = running_l2 / num_examples
    nll = running_nll / num_examples
    kl = running_kl / num_examples
    return l2, nll, kl


def train(
    model: DDP,
    ema: DINOForesightPCAVAE | DINORawVAE,
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
    best_val_l2: float,
    checkpoint_path: Path,
    samples_path: Path,
):
    model.train()
    ema.eval()  # EMA model should always be in eval mode

    # clean optimizer
    optimizer.zero_grad()
    step = initial_step

    match cfg.objective:
        case "AE":
            deterministic = True
        case _:
            deterministic = False

    if rank == 0:
        logger.info(
            f"deterministic: {deterministic}"
        )
    for epoch in range(initial_epoch, cfg.training.epochs):
        train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.training.mixed_precision):
                out = training_step(
                    model,
                    deterministic,
                    batch,
                    device
                )
                loss = -out.elbo.mean()
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
                step_kl = out.kl.mean()
                step_nll = -out.ll.mean()

                _, h, w, d = out.sample.shape
                step_l2 = (-out.ll / (h * w * d)).mean()

                logger.info(
                    f"[epoch={epoch:04d}, step={step:08d}] loss={step_loss:.8f}"
                )
                if cfg.training.wandb and rank == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "train_loss": step_loss,
                            "train_kl": step_kl,
                            "train_nll": step_nll,
                            "train_l2": step_l2,
                            "lr": current_lr
                        },
                        step=step
                    )

            step += 1

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
                    best_val_l2,
                    cfg
                )
                logger.info(f"saved checkpoint to {checkpoint_path}")

        if epoch % cfg.training.evaluate_every_epochs == 0:

            model.eval()
            torch.cuda.empty_cache()
            (l2, nll, kl) = evaluate(
                model.module,
                val_dataloader,
                deterministic,
                cfg.training.mixed_precision,
                device
            )
            torch.cuda.empty_cache()
            model.train()

            dist.all_reduce(l2, op=dist.ReduceOp.AVG)
            dist.all_reduce(nll, op=dist.ReduceOp.AVG)
            dist.all_reduce(kl, op=dist.ReduceOp.AVG)

            if rank == 0:
                logger.info(
                    f"[epoch={epoch}] l2: {l2.item()}"
                )
                logger.info(
                    f"[epoch={epoch}] nll: {nll.item()}")
                logger.info(
                    f"[epoch={epoch}] kl: {kl.item()}")
                if l2 < best_val_l2:
                    best_val_l2 = l2
                    save(
                        checkpoint_path / f"best.pt",
                        model.module,
                        ema,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        best_val_l2,
                        cfg
                    )
                    logger.info(f"saved checkpoint to {checkpoint_path}")

                if cfg.training.wandb:
                    wandb.log(
                        {"val_l2": l2.item()},
                        step=step
                    )
                    wandb.log(
                        {"val_nll": nll.item()},
                        step=step
                    )
                    wandb.log(
                        {"val_kl": kl.item()},
                        step=step
                    )

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
    best_val_l2 = pow(10, 9)
    # --- Model ---
    if "pca" in cfg.name:
        model = DINOForesightPCAVAE(**cfg.model)
    else:
        model = DINORawVAE(**cfg.model)
    # Create an EMA of the model for use after training
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    if rank == 0:
        summary(model.vae, depth=3)
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
        best_val_l2 = ckpt["best_val_l2"]

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
        best_val_l2,
        checkpoint_dir,
        samples_dir,
    )

    cleanup()


if __name__ == '__main__':
    main()
