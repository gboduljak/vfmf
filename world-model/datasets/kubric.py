
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, TypeVar

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from selection import dino_foresight_eval_selection
from transforms import transform_validation

A = TypeVar('A')


def deterministic_shuffle(items: List[A], seed: int = 42) -> List[A]:
  rng = random.Random(seed)
  shuffled = list(items)
  rng.shuffle(shuffled)
  return shuffled


class KubricDataset(Dataset):
  def __init__(
      self,
      dataset_root: str,
      split="train",
      suite="movi_a",
      original_resolution=256,
      transform: Callable[
          [List[Image.Image]],
          torch.Tensor
      ] = lambda x: x  # type: ignore
  ):
    super().__init__()
    self.data_path = dataset_root
    self.transform = transform
    self.sequences = []
    self.sequence_to_frames = {}
    seqs_dir = os.path.join(
        dataset_root,
        suite,
        f'{original_resolution}x{original_resolution}',
        split
    )
    self.num_frames = 0
    for seq in os.listdir(seqs_dir):
      frames = list(
          sorted([
              os.path.join(seqs_dir, seq, f)
              for f in os.listdir(os.path.join(seqs_dir, seq))
              if f.endswith("_rgb.png")
          ])
      )
      self.sequences.append(seq)
      self.sequence_to_frames[seq] = frames
      self.num_frames += len(frames)
    self.sequences = list(sorted(self.sequences))

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence_name = self.sequences[idx]
    frames_paths = self.sequence_to_frames[sequence_name]
    if self.transform:
      x = self.transform(frames_paths)
      return x, sequence_name
    else:
      return frames_paths, sequence_name


class KubricMultiDatasetFramesOnly(Dataset):
  def __init__(
      self,
      dataset_root: str,
      context_length: int,
      transform: Callable[
          [List[Image.Image]],
          torch.Tensor
      ] = lambda x: x,  # type: ignore
      max_size: Optional[int] = None
  ):
    super().__init__()
    root_dir = Path(dataset_root)

    self.scene_rollouts: Dict[Path, List[Path]] = {}

    for seed_dir in sorted(root_dir.iterdir()):
      for scene_dir in sorted(seed_dir.iterdir()):
        self.scene_rollouts[scene_dir] = list(sorted(scene_dir.iterdir()))

    if context_length != 1:
      # there is only one GT
      scene_rollouts: Dict[Path, List[Path]] = {}
      for (_, rollouts) in sorted(self.scene_rollouts.items()):
        for rollout in rollouts:
          scene_rollouts[rollout] = [rollout]
      self.scene_rollouts = scene_rollouts

    self.scenes = list(sorted(self.scene_rollouts.keys()))
    if max_size is not None and len(self.scenes) > max_size:
      self.scenes = deterministic_shuffle(self.scenes)
      self.scenes = self.scenes[:max_size]
    self.context_length = context_length
    self.transform = transform

  def __len__(self):
    return len(self.scenes)

  def __getitem__(self, idx):
    scene_path = self.scenes[idx]
    rollout, *_ = self.scene_rollouts[scene_path]
    frames_paths = list(
        sorted(
            p for p in rollout.iterdir() if "rgba" in p.name
        )
    )
    if self.context_length == 1:
      sequence_name = scene_path.stem
    else:
      sequence_name = f"{scene_path.parent.stem}_{scene_path.stem}"

    if self.transform:
      x = self.transform(frames_paths)  # type: ignore
      return x, sequence_name
    else:
      return frames_paths, sequence_name


class KubricExample(TypedDict):
  sequence_name: str
  context_rgb: torch.Tensor
  future_rgb: torch.Tensor
  context_segm: torch.Tensor
  context_depth: torch.Tensor
  context_surface_normals: torch.Tensor
  gt_segm: torch.Tensor
  gt_depth: torch.Tensor
  gt_surface_normals: torch.Tensor


class KubricMultiDataset(Dataset):
  def __init__(
      self,
      dataset_root: str,
      context_length: int = 4,
      resize_size: Tuple[int, int] = (224, 224),
      max_size: Optional[int] = None
  ):
    super().__init__()
    root_dir = Path(dataset_root)

    self.scene_rollouts: Dict[Path, List[Path]] = {}

    for seed_dir in sorted(root_dir.iterdir()):
      for scene_dir in sorted(seed_dir.iterdir()):
        self.scene_rollouts[scene_dir] = list(sorted(scene_dir.iterdir()))

    if context_length != 1:
      # there is only one GT
      scene_rollouts: Dict[Path, List[Path]] = {}
      for (_, rollouts) in sorted(self.scene_rollouts.items()):
        for rollout in rollouts:
          scene_rollouts[rollout] = [rollout]
      self.scene_rollouts = scene_rollouts

    self.scenes = list(sorted(self.scene_rollouts.keys()))
    if max_size is not None and len(self.scenes) > max_size:
      self.scenes = deterministic_shuffle(self.scenes)
      self.scenes = self.scenes[:max_size]
    self.context_length = context_length
    self.resize_size = resize_size

  def __len__(self):
    return len(self.scenes)

  def __getitem__(self, idx):
    scene_path = self.scenes[idx]
    
    # --- RGB Loading (Context & Future) ---
    # We use the first rollout for RGB context to determine paths
    rollout_ref, *_ = self.scene_rollouts[scene_path]
    frames_paths = list(
        sorted(
            p for p in rollout_ref.iterdir() if "rgba" in p.name
        )
    )
    
    # Selection logic
    select = dino_foresight_eval_selection(
        self.context_length,
        "uncertain",  # type: ignore
        "kubric"
    )  # type: ignore
    
    selection = select(frames_paths)  # type: ignore
    context_frames_paths = selection.context
    future_frames_paths = selection.future

    # RGB Transform (includes resizing)
    rgb_transform = transform_validation(
        self.resize_size,
        dinov2=True,
        convert_to_rgb=True
    )
    context_rgb = rgb_transform(
        [Image.open(f) for f in context_frames_paths]
    )
    future_rgb = rgb_transform(
        [Image.open(f) for f in future_frames_paths]
    )

    # --- Modality Loading (Context & GT) ---
    all_context_segms = []
    all_context_depths = []
    all_context_normals = []
    
    all_gt_segms = []
    all_gt_depths = []
    all_gt_normals = []

    # Define resizing transforms for additional modalities
    # Nearest for segmentation to preserve integer class labels
    resize_nearest = T.Resize(self.resize_size, interpolation=T.InterpolationMode.NEAREST)
    # Bilinear for continuous values (Depth/Normals)
    resize_bilinear = T.Resize(self.resize_size, interpolation=T.InterpolationMode.BILINEAR)

    for rollout in self.scene_rollouts[scene_path]:
      with open(rollout / "data_ranges.json", "rb") as fs:
        data_ranges = json.load(fs)

      # ---------------------------
      # 1. Segmentation
      # ---------------------------
      segm_paths = [
          rollout / f"segmentation_{frame_idx:05d}.png"
          for frame_idx in range(0, 24)
      ]
      sel_segm = select(segm_paths)
      
      # Helper to process segmentation
      def process_segm(path):
          s = np.array(Image.open(path))
          s = s > 0
          s_pil = Image.fromarray(s, "L")
          # Convert to tensor and Resize
          return resize_nearest(T.PILToTensor()(s_pil))

      # Load Context Segmentation
      curr_ctx_segm = torch.stack([process_segm(p) for p in sel_segm.context])
      all_context_segms.append(curr_ctx_segm)

      # Load GT Segmentation (Future)
      # Existing logic: only takes the LAST future frame
      gt_segm_path = sel_segm.future[-1]
      all_gt_segms.append(process_segm(gt_segm_path))

      # ---------------------------
      # 2. Depth
      # ---------------------------
      depth_paths = [
          rollout / f"depth_{frame_idx:05d}.png"
          for frame_idx in range(0, 24)
      ]
      sel_depth = select(depth_paths)

      # Helper to process depth
      def process_depth(path):
          d = np.array(Image.open(path)) # type: ignore
          min_d, max_d = (data_ranges["depth"]["min"], data_ranges["depth"]["max"])
          
          # Normalize logic provided in original code
          d_normalized = d.astype(np.float32) / 65535.0
          d_actual = min_d + d_normalized * (max_d - min_d)
          d_norm = (d_actual - min_d) / (max_d - min_d)
          d_norm = np.clip(d_norm, 0.0, 1.0)
          d_8bit = (d_norm * 255).astype(np.uint8)
          
          d_pil = Image.fromarray(d_8bit, "L")
          return resize_bilinear(T.PILToTensor()(d_pil))

      # Load Context Depth
      curr_ctx_depth = torch.stack([process_depth(p) for p in sel_depth.context])
      all_context_depths.append(curr_ctx_depth)

      # Load GT Depth
      gt_depth_path = sel_depth.future[-1]
      all_gt_depths.append(process_depth(gt_depth_path))

      # ---------------------------
      # 3. Surface Normals
      # ---------------------------
      normals_paths = [
          rollout / f"normal_{frame_idx:05d}.png"
          for frame_idx in range(0, 24)
      ]
      sel_normals = select(normals_paths)

      # Helper to process normals
      def process_normal(path):
          n = np.array(Image.open(path)) # type: ignore
          n = torch.from_numpy(n).permute(2, 0, 1)
          n = n / 65535.0
          # Resize requires [C, H, W]
          return resize_bilinear(n)

      # Load Context Normals
      curr_ctx_norm = torch.stack([process_normal(p) for p in sel_normals.context])
      all_context_normals.append(curr_ctx_norm)

      # Load GT Normals
      gt_normal_path = sel_normals.future[-1]
      all_gt_normals.append(process_normal(gt_normal_path))

    if self.context_length == 1:
      sequence_name = scene_path.stem
    else:
      sequence_name = f"{scene_path.parent.stem}_{scene_path.stem}"

    # Stack results. 
    # Note: 'all_gt_*' is stacked across rollouts (usually just 1).
    # 'all_context_*' is a list of (T, C, H, W) tensors. We stack them to be (Batch, T, C, H, W) or squeeze if needed.
    # To match 'gt_segm' behavior which is simply stacked, we do the same.
    # If context_length != 1, len(all_context_segms) is 1. The result will be (1, T, C, H, W).
    # If the downstream expects (T, C, H, W), we might need to squeeze, but strict adherence to stacking follows the existing pattern.
    
    # However, existing context_rgb is (T, C, H, W).
    # Since we iterated rollouts, let's assume we want the first element if len==1 to match context_rgb
    # BUT `all_gt_segms` is torch.stack-ed.
    
    # We will simply torch.stack them. If there is 1 rollout, dimension 0 is size 1.
    # If the user wants strictly (T, C, H, W) for context, they should modify the rollout loop logic or squeeze later.
    # Based on `context_rgb` being distinct from the loop, we'll return the first rollout's context to match context_rgb shape exactly.
    
    final_context_segm = all_context_segms[0]
    final_context_depth = all_context_depths[0]
    final_context_normal = all_context_normals[0]

    return KubricExample(
        sequence_name=sequence_name,
        context_rgb=context_rgb,
        future_rgb=future_rgb,
        context_segm=final_context_segm,
        context_depth=final_context_depth,
        context_surface_normals=final_context_normal,
        gt_segm=torch.stack(all_gt_segms),
        gt_depth=torch.stack(all_gt_depths),
        gt_surface_normals=torch.stack(all_gt_normals)
    )