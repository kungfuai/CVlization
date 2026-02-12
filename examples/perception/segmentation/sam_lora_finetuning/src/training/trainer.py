"""Trainer for SAM LoRA fine-tuning."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models.sam_forward import _batch_to_device, sam_forward


class Trainer:
    """Encapsulates training, validation, and visualization for SAM LoRA."""

    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader) -> list[float]:
        """Run one training epoch. Returns list of per-batch losses."""
        self.model.train()
        losses = []
        for batch in tqdm(dataloader, desc="  train"):
            batch = _batch_to_device(batch, self.device)
            outputs = sam_forward(self.model, batch, multimask_output=False)

            gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
            pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
            pred = pred.squeeze(1)            # [B, 1, 1, H, W] -> [B, 1, H, W]
            gt = gt.unsqueeze(1).float().to(self.device)  # [B, H, W] -> [B, 1, H, W]

            loss = self.loss_fn(pred, gt)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    @torch.no_grad()
    def validate_epoch(self, dataloader) -> list[float]:
        """Run validation. Returns list of per-batch losses."""
        self.model.eval()
        losses = []
        for batch in tqdm(dataloader, desc="  val"):
            batch = _batch_to_device(batch, self.device)
            outputs = sam_forward(self.model, batch, multimask_output=False)

            gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
            pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
            pred = pred.squeeze(1)
            gt = gt.unsqueeze(1).float().to(self.device)

            loss = self.loss_fn(pred, gt)
            losses.append(loss.item())
        return losses

    @torch.no_grad()
    def visualize_predictions(self, dataset, max_samples=4, target_h=256) -> list:
        """Run inference on validation samples and build side-by-side images.

        Returns a list of numpy arrays, each showing:
            [input image | GT mask | predicted mask]
        with a white margin between panels.  All canvases are normalised to
        ``target_h`` so wandb doesn't warn about mismatched sizes.
        """
        from PIL import Image as PILImage

        self.model.eval()
        margin = 10  # pixels between panels
        panels = []

        def _resize(arr, th, tw):
            return np.array(PILImage.fromarray(arr).resize((tw, th), PILImage.BILINEAR))

        n = min(max_samples, len(dataset))
        for i in range(n):
            sample = dataset[i]

            # --- original-resolution image (re-read from dataset) ---
            if hasattr(dataset, "samples"):
                entry = dataset.samples[i]
                if isinstance(entry, tuple) and isinstance(entry[0], Path):
                    # ImageMaskDataset: (img_path, mask_path)
                    orig_img = np.array(PILImage.open(entry[0]).convert("RGB"))
                else:
                    # CocoSamDataset: (img_info dict, ann)
                    img_info = entry[0]
                    orig_img = np.array(
                        PILImage.open(dataset.images_dir / img_info["file_name"]).convert("RGB")
                    )
            else:
                orig_img = sample["image"].permute(1, 2, 0).numpy().astype(np.uint8)

            h, w = sample["original_size"]

            # --- GT mask -> RGB (green channel) ---
            gt_mask = sample["ground_truth_mask"].numpy()  # (H, W) binary
            gt_rgb = np.stack(
                [np.zeros_like(gt_mask), gt_mask * 255, np.zeros_like(gt_mask)], axis=-1
            ).astype(np.uint8)

            # --- predicted mask (already at original resolution) ---
            batch = [
                {
                    "image": sample["image"].to(self.device),
                    "boxes": sample["boxes"].to(self.device),
                    "original_size": sample["original_size"],
                }
            ]
            outputs = sam_forward(self.model, batch, multimask_output=False)
            pred_logits = outputs[0]["low_res_logits"]  # (1, 1, H, W) at original res
            pred_mask = (pred_logits.squeeze().cpu().numpy() > 0).astype(np.uint8)
            pred_rgb = np.stack(
                [np.zeros_like(pred_mask), pred_mask * 255, np.zeros_like(pred_mask)], axis=-1
            ).astype(np.uint8)

            # Resize all three panels to uniform target_h (preserve aspect ratio)
            scale = target_h / h
            target_w = int(w * scale)
            img_panel = _resize(orig_img, target_h, target_w)
            gt_panel = _resize(gt_rgb, target_h, target_w)
            pred_panel = _resize(pred_rgb, target_h, target_w)

            # Build side-by-side with white margin
            sep = np.full((target_h, margin, 3), 255, dtype=np.uint8)
            canvas = np.concatenate([img_panel, sep, gt_panel, sep, pred_panel], axis=1)
            panels.append(canvas)

        return panels
