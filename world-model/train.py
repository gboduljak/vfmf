import os
from copy import deepcopy
from functools import partial
from logging import Logger
from pathlib import Path
from typing import OrderedDict

import einops
import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdiffeq import odeint
from torchinfo import summary
from torchvision.transforms import Compose

from datasets.cityscapes import Cityscapes, CityscapesFrameDataset
from datasets.kubric import KubricDataset
from ddp import cleanup, setup_ddp
from log import create_logger
from models.autoregressive_dino_foresight import AutoregressiveDINOForesight
from models.autoregressive_dino_foresight_latent import \
    AutoregressiveDINOForesightLatent
from models.autoregressive_dino_foresight_pca import \
    AutoregressiveDINOForesightPCA
from recipe import (load_frames, sample_train_dino_foresight,
                    sample_train_dino_foresight_variable_length,
                    sample_val_dino_foresight)
from scheduler import LinearWarmupScheduler
from seed import seed_everything, worker_init_function
from transforms import transform_train
from visualization import visualize_dino_features_pca_single_pil

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def save(
    path: Path,
    model: AutoregressiveDINOForesightPCA,
    ema: AutoregressiveDINOForesightPCA,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | LinearWarmupScheduler,
    epoch: int,
    step: int,
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
      **wandb_info
  }, path)


def training_step(
        velocity_module: AutoregressiveDINOForesightPCA | AutoregressiveDINOForesight | AutoregressiveDINOForesightLatent,
        batch,
        mixed_precision,
        recipe,
        min_sequence_length,
        max_sequence_length,
        device):
  x, *_ = batch
  x = x.to(device)
  match recipe:
    case "dino_foresight_variable_length":
      t = np.random.randint(min_sequence_length, max_sequence_length + 1)
    case _:
      t = max_sequence_length
  x = x[:, -t:, ...]
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
    # x: [b, t, c, H, W]
    with torch.inference_mode():
      if isinstance(velocity_module.module, AutoregressiveDINOForesightLatent):
        f = velocity_module.module.encode(x)
      else:
        f = velocity_module.module.preprocess(x)
    # f: [b, t, h, w, d]
    b, t, h, w, d = f.shape
    # Split into context and target
    f_ctx, f_tgt = (
        f[:, :-1, ...],
        f[:, -1:, ...]
    )
    # Sample time
    t = torch.rand(size=[b, ], device=f.device)
    # Construct target
    x1 = f_tgt
    x0 = torch.randn_like(x1)
    xt = (
        (1 - t).view((b, 1, 1, 1, 1)) * x0 +
        t.view((b, 1, 1, 1, 1)) * x1
    )
    # Regress the target velocity
    v = velocity_module(f_ctx, xt, t)
    # Compute loss
    loss = F.mse_loss(v, x1 - x0)
    return loss


def evaluation_step(velocity, batch, device):
  x, *_ = batch
  num_samples = 8
  num_steps = 10
  x = x.to(device)
  # x: [b, t, c, H, W]
  with torch.inference_mode():
    if isinstance(velocity, AutoregressiveDINOForesightLatent):
      f = velocity.encode(x)
    else:
      f = velocity.preprocess(x)
  # f: [b, t, h, w, d]
  b, t, h, w, d = f.shape
  # Split into context and target
  f_ctx = f[:, :-1, ...]
  f_tgt = f[:, [-1], ...]
  # Estimate distribution mean
  f_preds = []
  for k in range(num_samples):
    f_pred, _ = sample(
        velocity,
        f_ctx,
        num_steps
    )
    # if isinstance(velocity, AutoregressiveDINOForesightLatent):
    #   f_pred = velocity.decode(f_pred)
    #   f_tgt = velocity.decode(f_tgt)
    f_preds.append(f_pred)
  f_preds = torch.stack(f_preds)
  f_preds = einops.rearrange(
      f_preds,
      "k b 1 h w d -> b k 1 h w d"
  )
  f_mean_pred = torch.mean(
      f_preds,
      dim=1
  )
  mean_future_mse = F.mse_loss(f_mean_pred, f_tgt)
  # Estimate best of K
  f_tgt_rep = einops.repeat(
      f_tgt,
      "b 1 h w d -> b k 1 h w d",
      k=num_samples
  )
  min_of_k_mse = (
      F.mse_loss(
          f_preds,
          f_tgt_rep,
          reduction="none"
      )
      .mean(dim=[2, 3, 4, 5])
      .min(dim=1).values
      .mean(dim=0)
  )
  mean_of_k_mse = (
      F.mse_loss(
          f_preds,
          f_tgt_rep,
          reduction="none"
      )
      .mean(dim=[2, 3, 4, 5])
      .mean(dim=1)
      .mean(dim=0)
  )
  return (mean_future_mse, min_of_k_mse, mean_of_k_mse)


@torch.inference_mode()
def sample(
    velocity: AutoregressiveDINOForesightPCA,
    f_ctx,
    steps=100,
    method="euler"
):
  # f_ctx: [b, t - 1, h, w, d]
  b, _, h, w, d = f_ctx.shape
  f0 = torch.randn((b, 1, h, w, d), device=f_ctx.device)
  t = torch.linspace(0., 1., steps, device=f_ctx.device)
  # Define ODE

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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
  ema_params = OrderedDict(ema_model.named_parameters())
  model_params = OrderedDict(model.named_parameters())
  for name, param in model_params.items():
    ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def evaluate(
    model: AutoregressiveDINOForesightPCA,
    val_dataloader: DataLoader,
    device: torch.device
):
  num_examples = 0
  (running_future_mse, running_min_of_k_mse, running_mean_of_k_mse) = (0, 0, 0)
  for batch in val_dataloader:
    x, *_ = batch
    b, *_ = x.shape
    (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluation_step(
        model,
        batch,
        device
    )
    running_future_mse += mean_future_mse * b
    running_min_of_k_mse += min_of_k_mse * b
    running_mean_of_k_mse += mean_of_k_mse * b
    num_examples += b

  # Calculate the final averages by dividing the running sums by the total
  # number of examples
  final_future_mse = running_future_mse / num_examples
  final_min_of_k_mse = running_min_of_k_mse / num_examples
  final_mean_of_k_mse = running_mean_of_k_mse / num_examples

  return final_future_mse, final_min_of_k_mse, final_mean_of_k_mse


@torch.inference_mode()
def plot_samples(
    model,
    batch,
    device
):
  num_samples = 4
  num_steps = 10
  x, *_ = batch
  x = x.to(device)
  if isinstance(model, AutoregressiveDINOForesightLatent):
    f = model.encode(x)
  else:
    f = model.preprocess(x)
  # f: [b, t, h, w, d]
  b, t, h, w, d = f.shape
  # Split into context and target
  f_ctx = f[:, :-1, ...]
  f_tgt = f[:, [-1], ...]
  # Estimate distribution mean
  f_preds = []
  for _ in range(num_samples):
    f_pred, _ = sample(
        model,
        f_ctx,
        num_steps
    )
    f_preds.append(f_pred)
  f_preds = torch.stack(f_preds)
  f_preds = einops.rearrange(
      f_preds,
      "k b 1 h w d -> b k 1 h w d"
  )
  f_mean_pred = torch.mean(
      f_preds,
      dim=1
  )
  if isinstance(model, AutoregressiveDINOForesightLatent):
    x_ctx = f_ctx
    x_tgt = f_tgt
    x_mean_pred = f_mean_pred
    x_preds = einops.rearrange(
        einops.rearrange(
            f_preds,
            "b k 1 h w d -> (b k) 1 h w d",
            k=num_samples
        ),
        "(b k) 1 h w d -> b k h w d",
        k=num_samples
    )
  else:
    f = model.preprocess(x)
    x_ctx = model.postprocess(f_ctx)
    x_tgt = model.postprocess(f_tgt)
    x_mean_pred = model.postprocess(f_mean_pred)
    x_preds = einops.rearrange(
        model.postprocess(
            einops.rearrange(
                f_preds,
                "b k 1 h w d -> (b k) 1 h w d",
                k=num_samples
            )
        ),
        "(b k) 1 h w d -> b k h w d",
        k=num_samples
    )
  return visualize_dino_features_pca_single_pil(
      x_ctx.cpu(),
      x_tgt.cpu(),
      x_mean_pred.cpu(),
      x_preds.cpu()
  )


def train(
    model: DDP,
    ema: AutoregressiveDINOForesightPCA,
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
    checkpoint_path: Path,
    samples_path: Path,
    recipe: str,
    min_sequence_length: int,
    max_sequence_length: int
):
  model.train()
  ema.eval()  # EMA model should always be in eval mode

  optimizer.zero_grad()

  step = initial_step

  for epoch in range(initial_epoch, cfg.training.epochs):
    train_sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(train_dataloader):
      match cfg.objective:
        case "CFM":
          loss = training_step(
              model,
              batch,
              cfg.training.mixed_precision,
              recipe,
              min_sequence_length,
              max_sequence_length,
              device)
        case _:
          raise NotImplementedError()

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
            f"[epoch={epoch:04d}, step={step:08d}] loss={step_loss:.8f}"
        )
        if cfg.training.wandb and rank == 0:
          current_lr = optimizer.param_groups[0]["lr"]
          wandb.log({"train_loss": step_loss,
                    "lr": current_lr}, step=step)

      step += 1

    if rank == 0:
      save(
          checkpoint_path / f"current.pt",
          model.module,
          ema,
          optimizer,
          scheduler,
          epoch,
          step,
          cfg
      )
      logger.info(f"saved checkpoint to {checkpoint_path}")

    dist.barrier()

    if epoch % cfg.training.ckpt_every_epochs == 0:
      if rank == 0:
        save(
            checkpoint_path / f"epoch={epoch:04d}.pt",
            model.module,
            ema,
            optimizer,
            scheduler,
            epoch,
            step,
            cfg
        )
        logger.info(f"saved checkpoint to {checkpoint_path}")

    dist.barrier()

    if epoch % cfg.training.evaluate_every_epochs == 0:

      model.eval()
      (mean_future_mse, min_of_k_mse, mean_of_k_mse) = evaluate(
          model.module,
          val_dataloader,
          device
      )
      model.train()

      dist.all_reduce(mean_future_mse, op=dist.ReduceOp.AVG)
      dist.all_reduce(min_of_k_mse, op=dist.ReduceOp.AVG)
      dist.all_reduce(mean_of_k_mse, op=dist.ReduceOp.AVG)

      if rank == 0:
        logger.info(
            f"[epoch={epoch}] mean_future_mse: {mean_future_mse.item()}"
        )
        logger.info(
            f"[epoch={epoch}] min_of_k_mse: {min_of_k_mse.item()}")
        logger.info(
            f"[epoch={epoch}] mean_of_k_mse: {mean_of_k_mse.item()}")
        if cfg.training.wandb:
          wandb.log(
              {"val_mean_future_mse": mean_future_mse.item()}, step=step)
          wandb.log(
              {"val_min_of_k_mse": min_of_k_mse.item()}, step=step)
          wandb.log(
              {"val_mean_of_k_mse": mean_of_k_mse.item()}, step=step)

    dist.barrier()

    if epoch % cfg.training.sample_every_epochs == 0:
      for batch in val_dataloader:
        break
      samples = plot_samples(model.module, batch, device)  # type: ignore
      (samples_path / f"rank={rank}").mkdir(exist_ok=True, parents=True)
      samples.save((samples_path / f"rank={rank}" / f"{step}.jpg"))
      if cfg.training.wandb and rank == 0:
        wandb.log({"val_samples": wandb.Image(samples)})

    dist.barrier()


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
  # --- Model ---
  if "pca" in cfg.name:
    model = AutoregressiveDINOForesightPCA(**cfg.model)
  elif "vae" in cfg:
    model = AutoregressiveDINOForesightLatent(**cfg.model, vae=cfg.vae)
  else:
    model = AutoregressiveDINOForesight(**cfg.model)
  # Create an EMA of the model for use after training
  ema = deepcopy(model).to(device)
  for p in ema.parameters():
    p.requires_grad = False
  if rank == 0:
    summary(model.transformer, depth=3)
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
  if cfg.training.wandb:
    if "resume" in cfg and cfg.resume.wandb:
      if rank == 0:
        wandb.init(
            project="vgg-wm",
            config=OmegaConf.to_container(
                cfg, resolve=True),  # type: ignore
            resume="must",
            id=ckpt["wandb"]["run"]["id"]
        )
    else:
      if rank == 0:
        wandb.init(
            project="vgg-wm",
            config=OmegaConf.to_container(
                cfg, resolve=True),  # type: ignore
        )
  # Wrap in DDP
  model = model.to(device)
  ema = ema.to(device)
  model = DDP(model)
  # --- Create Dataset and Sampler ---
  recipe = cfg.recipe.name
  match recipe:
    case "dino_foresight":
      train_transform = Compose([
          sample_train_dino_foresight(
              sequence_length=cfg.model.sequence_length
          ),
          load_frames(),
          transform_train(**cfg.transforms.train),
      ])
      min_sequence_length = cfg.model.sequence_length
      max_sequence_length = cfg.model.sequence_length
    case _:
      min_sequence_length = cfg.recipe.min_sequence_length
      max_sequence_length = cfg.recipe.max_sequence_length
      train_transform = Compose([
          sample_train_dino_foresight_variable_length(
              max_sequence_length=cfg.recipe.max_sequence_length
          ),
          load_frames(),
          transform_train(**cfg.transforms.train)
      ])
  val_transform = Compose([
      sample_val_dino_foresight(sequence_length=cfg.model.sequence_length),
      load_frames(),
      transform_train(**cfg.transforms.validation)
  ])
  if "cityscapes" in cfg.data.dataset_root:
    train_dataset = Cityscapes(
        **cfg.data,
        transform=train_transform,
        split="train",
    )
    val_dataset = Cityscapes(
        **cfg.data,
        transform=val_transform,
        split="validation",
    )
  else:
    train_dataset = KubricDataset(
        **cfg.data,
        transform=train_transform,
        split="train"
    )
    val_dataset = KubricDataset(
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
      drop_last=True
  )
  val_dataloader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      sampler=val_sampler,
      num_workers=cfg.training.num_workers,
      worker_init_fn=partial(worker_init_function, global_rank=rank)
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
      checkpoint_dir,
      samples_dir,
      recipe,
      min_sequence_length,
      max_sequence_length
  )

  cleanup()


if __name__ == '__main__':
  main()
