from typing import List

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

from transforms import denormalize_dino


def visualize_dino_features_pca_single_pil(
    x_ctx,
    x_tgt,
    x_mean_pred,
    x_preds,
    output_size=(256, 256)
):
  """
  Performs PCA on DINO features, upsamples the 3D projection, and renders
  all plots onto a single PIL.Image object with titles.

  Args:
      x_ctx (torch.Tensor): Context features, shape (B, C, H, W, D)
      x_tgt (torch.Tensor): Target features, shape (B, 1, H, W, D)
      x_mean_pred (torch.Tensor): Mean prediction, shape (B, 1, H, W, D)
      x_preds (torch.Tensor): Predictions, shape (B, C_preds, H, W, D)
      output_size (tuple): The target (width, height) for each upsampled image.

  Returns:
      PIL.Image.Image: A single PIL Image containing the entire plot.
  """
  B, C_ctx, H, W, D = x_ctx.shape
  C_preds = x_preds.shape[1]

  # Titles for the columns
  col_titles = (
      [f"Context {i}" for i in range(C_ctx)] +
      ["Target"] +
      ["Mean Pred"] +
      [f"Pred {i}" for i in range(C_preds)]
  )
  num_cols = len(col_titles)

  # Calculate dimensions for the final image
  title_height = 20  # Height for the title row
  total_width = num_cols * output_size[0]
  total_height = B * (output_size[1] + title_height)

  # Create a blank image with a white background
  final_image = Image.new('RGB', (total_width, total_height), 'white')
  draw = ImageDraw.Draw(final_image)

  try:
    font = ImageFont.truetype("arial.ttf", 10)
  except IOError:
    font = ImageFont.load_default()

  for batch_idx in range(B):
    # Flatten all features for the current batch
    ctx_flat = x_ctx[batch_idx].view(-1, D)
    tgt_flat = x_tgt[batch_idx].view(-1, D)
    mean_pred_flat = x_mean_pred[batch_idx].view(-1, D)
    preds_flat = x_preds[batch_idx].view(-1, D)

    all_features = torch.cat(
        [ctx_flat, tgt_flat, mean_pred_flat, preds_flat], dim=0
    ).detach().cpu().numpy()

    # Fit PCA and transform the features
    pca = PCA(n_components=3)
    pca.fit(all_features)

    ctx_pca = pca.transform(ctx_flat).reshape(C_ctx, H, W, 3)
    tgt_pca = pca.transform(tgt_flat).reshape(1, H, W, 3)
    mean_pred_pca = pca.transform(mean_pred_flat).reshape(1, H, W, 3)
    preds_pca = pca.transform(preds_flat).reshape(C_preds, H, W, 3)

    # Normalize PCA output to [0, 255]
    def normalize_pca(data):
      min_val = data.min()
      max_val = data.max()
      normalized = (data - min_val) / (max_val - min_val)
      return (normalized * 255).astype(np.uint8)

    ctx_pca = normalize_pca(ctx_pca)
    tgt_pca = normalize_pca(tgt_pca)
    mean_pred_pca = normalize_pca(mean_pred_pca)
    preds_pca = normalize_pca(preds_pca)

    all_images_np = (
        list(ctx_pca) +
        list(tgt_pca) +
        list(mean_pred_pca) +
        list(preds_pca)
    )

    # Plotting logic for the current batch
    for i, img_np in enumerate(all_images_np):
      # Create a PIL image from the numpy array and upsample it
      img_pil = Image.fromarray(img_np)
      img_upsampled = img_pil.resize(output_size, Image.BILINEAR)

      # Calculate paste position
      x_offset = i * output_size[0]
      y_offset = batch_idx * (output_size[1] + title_height)

      # Paste the image onto the final image
      final_image.paste(
          img_upsampled, (x_offset, y_offset + title_height))

      # Draw the title using textbbox
      title_text = col_titles[i]
      # Get the bounding box of the text
      bbox = draw.textbbox((0, 0), title_text, font=font)
      text_width = bbox[2] - bbox[0]
      text_height = bbox[3] - bbox[1]

      title_x = x_offset + (output_size[0] - text_width) / 2
      title_y = y_offset + (title_height - text_height) / 2
      draw.text((title_x, title_y), title_text, fill="black", font=font)

  return final_image



@torch.no_grad()
def decode_rgb(decoder: nn.Module, f: torch.Tensor, device: torch.device):
    import cv2
    import numpy as np
    import torchvision.transforms.functional as F

    f = f.to(device)
    rgb = decoder.decode(f, normalize=True) # type: ignore

    transform = denormalize_dino()
    images: List[Image.Image] = []

    for idx in range(f.shape[0]):
        # Denormalize and convert to CPU numpy RGB (H, W, 3)
        img = transform(rgb[idx, 0]).clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.bilateralFilter(
          img_np,
          d=3,
          sigmaColor=30,
          sigmaSpace=10
        )
        images.append(F.to_pil_image(img_np))

    return images


def segs_to_image(
    segmentation: torch.Tensor,
    colormap_name: str = 'viridis',
    color_map: dict = None # type: ignore
) -> Image.Image:
    """
    Converts a 2D segmentation tensor into a PIL Image.
    
    Args:
        segmentation: 2D torch.Tensor (H, W)
        colormap_name: str, Matplotlib colormap or 'bw'/'binary' for black & white.
        color_map: dict, Optional custom mapping {label: (r, g, b)}
    """
    if segmentation.ndim != 2:
        raise ValueError("Segmentation tensor must be 2D (H, W)")

    seg_np = segmentation.cpu().numpy() # Ensure on CPU
    h, w = seg_np.shape

    if color_map is not None:
        # Use custom color map dictionary
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in color_map.items():
            rgb[seg_np == label] = color
            
    elif colormap_name in ['bw', 'binary', 'black_white']:
        # Strict Black and White mode for binary segmentation
        # 0 -> Black, Non-zero -> White
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mask = seg_np > 0 
        rgb[mask] = [255, 255, 255]
        
    else:
        # Use matplotlib colormap with dynamic normalization
        cmap = cm.get_cmap(colormap_name)
        
        # Avoid division by zero if max is 0
        max_val = seg_np.max()
        if max_val == 0:
            normalized = seg_np.astype(float)
        else:
            normalized = seg_np.astype(float) / max_val
            
        rgb_float = cmap(normalized)[..., :3]  # drop alpha
        rgb = (rgb_float * 255).astype(np.uint8)

    return Image.fromarray(rgb)

def depth_to_image(depth: torch.Tensor, colormap=True) -> Image.Image:
  assert depth.dtype == torch.uint8, "Depth tensor must be uint8"

  depth_np = depth.cpu().numpy()

  if colormap:
    depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_TURBO)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(depth_color)
  else:
    return Image.fromarray(depth_np)

def normals_to_image(normals: torch.Tensor) -> Image.Image:
  assert normals.ndim == 3 and normals.shape[0] == 3, "Input must be (H,W,3)"
  normals_rgb = F.normalize(
      normals.permute(1, 2, 0),
      dim=-1,
      p=2
  )
  # Map [-1,1] -> [0,255]
  normals_rgb = (
      ((normals_rgb + 1.0) * 0.5 * 255.0)
      .clamp(0, 255)
      .byte()
      .cpu()
      .numpy()
  )
  return Image.fromarray(normals_rgb)