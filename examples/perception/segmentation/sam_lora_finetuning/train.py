#!/usr/bin/env python3
"""SAM LoRA Fine-tuning — main entry point.

Trains SAM ViT-B with LoRA adapters on a COCO-format segmentation dataset
using DiceCELoss from MONAI for stable, NaN-free training.
"""

import argparse
import sys
from pathlib import Path
from statistics import mean

import monai
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CocoSamDataset,
    ImageMaskDataset,
    collate_fn,
    download_hf_dataset,
    download_ring_dataset,
)
from lora import LoRA_sam


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune SAM ViT-B with LoRA adapters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hf-dataset", type=str, default=None, help="HuggingFace dataset to download")
    p.add_argument("--dataset-dir", type=str, default=None, help="Path to COCO dataset root")
    p.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rank", type=int, default=512, help="LoRA rank")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint", type=str, default=None, help="SAM ViT-B checkpoint path")
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="sam-lora-finetuning")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _batch_to_device(batch, device):
    """Move image and box tensors in a SAM batch to the given device."""
    for item in batch:
        item["image"] = item["image"].to(device)
        item["boxes"] = item["boxes"].to(device)
    return batch


def sam_forward(model, batched_input, multimask_output=False):
    """SAM forward pass WITHOUT @torch.no_grad() so gradients can flow.

    The pip-installed segment_anything decorates Sam.forward with
    @torch.no_grad(), which blocks training.  This reimplements the same
    logic with gradients enabled.

    Critically, the predicted masks are **post-processed** (upsampled) to the
    original image resolution *before* being returned — matching the behaviour
    of the original Sam_LoRA vendored SAM where the loss is computed at full
    resolution rather than the decoder's native 256x256.
    """
    input_images = torch.stack(
        [model.preprocess(x["image"]) for x in batched_input], dim=0
    )
    image_embeddings = model.image_encoder(input_images)

    outputs = []
    for image_record, curr_embedding in zip(batched_input, image_embeddings):
        points = None
        if "point_coords" in image_record:
            points = (image_record["point_coords"], image_record["point_labels"])

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=image_record.get("boxes", None),
            masks=image_record.get("mask_inputs", None),
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upsample to original image resolution (matches original Sam_LoRA repo)
        masks = model.postprocess_masks(
            low_res_masks,
            input_size=image_record["image"].shape[-2:],
            original_size=image_record["original_size"],
        )
        outputs.append({"low_res_logits": masks, "iou_predictions": iou_predictions})
    return outputs


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Run one training epoch. Returns list of per-batch losses."""
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="  train"):
        batch = _batch_to_device(batch, device)
        outputs = sam_forward(model, batch, multimask_output=False)

        # Both pred and GT are at original image resolution
        gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
        pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        pred = pred.squeeze(1)            # [B, 1, 1, H, W] -> [B, 1, H, W]
        gt = gt.unsqueeze(1).float().to(device)  # [B, H, W] -> [B, 1, H, W]

        loss = loss_fn(pred, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Run validation. Returns list of per-batch losses."""
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="  val"):
        batch = _batch_to_device(batch, device)
        outputs = sam_forward(model, batch, multimask_output=False)

        gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
        pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        pred = pred.squeeze(1)
        gt = gt.unsqueeze(1).float().to(device)

        loss = loss_fn(pred, gt)
        losses.append(loss.item())
    return losses


@torch.no_grad()
def visualize_predictions(model, dataset, device, max_samples=4, target_h=256):
    """Run inference on validation samples and build side-by-side images.

    Returns a list of numpy arrays, each showing:
        [input image | GT mask | predicted mask]
    with a white margin between panels.  All canvases are normalised to
    ``target_h`` so wandb doesn't warn about mismatched sizes.
    """
    from PIL import Image as PILImage

    model.eval()
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

        # --- GT mask → RGB (green channel) ---
        gt_mask = sample["ground_truth_mask"].numpy()  # (H, W) binary
        gt_rgb = np.stack([np.zeros_like(gt_mask), gt_mask * 255, np.zeros_like(gt_mask)], axis=-1).astype(np.uint8)

        # --- predicted mask (already at original resolution) ---
        batch = [{"image": sample["image"].to(device), "boxes": sample["boxes"].to(device),
                  "original_size": sample["original_size"]}]
        outputs = sam_forward(model, batch, multimask_output=False)
        pred_logits = outputs[0]["low_res_logits"]  # (1, 1, H, W) at original res
        pred_mask = (pred_logits.squeeze().cpu().numpy() > 0).astype(np.uint8)
        pred_rgb = np.stack([np.zeros_like(pred_mask), pred_mask * 255, np.zeros_like(pred_mask)], axis=-1).astype(np.uint8)

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


def main():
    args = parse_args()

    print("=" * 70)
    print("SAM LoRA Fine-tuning")
    print("=" * 70)

    # ---- 1. Dataset ----
    dataset_format = None  # "coco" or "image_mask"
    if args.hf_dataset:
        print(f"\n[1/4] Downloading dataset: {args.hf_dataset}")
        args.dataset_dir = download_hf_dataset(args.hf_dataset)
        dataset_format = "coco"
    elif args.dataset_dir is None:
        print("\n[1/4] No dataset specified — downloading default ring dataset...")
        args.dataset_dir = download_ring_dataset()
    else:
        print(f"\n[1/4] Using dataset: {args.dataset_dir}")

    dataset_path = Path(args.dataset_dir)

    # Auto-detect format if not already set
    if dataset_format is None:
        train_json = dataset_path / "train" / "_annotations.coco.json"
        if train_json.exists():
            dataset_format = "coco"
        elif (dataset_path / "train" / "images").is_dir() and (dataset_path / "train" / "masks").is_dir():
            dataset_format = "image_mask"
        else:
            print(f"ERROR: Cannot detect dataset format in {dataset_path}.")
            print("  Expected either train/_annotations.coco.json (COCO)")
            print("  or train/images/ + train/masks/ (image+mask)")
            sys.exit(1)

    # ---- 2. Model ----
    print("\n[2/4] Loading SAM ViT-B and applying LoRA...")
    from segment_anything import build_sam_vit_b

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint:
        sam = build_sam_vit_b(checkpoint=args.checkpoint)
    else:
        # Download default checkpoint
        ckpt_path = Path(args.output_dir) / "sam_vit_b_01ec64.pth"
        if not ckpt_path.exists():
            print("  Downloading SAM ViT-B checkpoint...")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, str(ckpt_path))
            print(f"  Saved to {ckpt_path}")
        sam = build_sam_vit_b(checkpoint=str(ckpt_path))

    sam_lora = LoRA_sam(sam, args.rank)
    model = sam_lora.sam
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  LoRA rank: {args.rank}")

    # ---- 3. Data loaders ----
    print(f"\n[3/4] Building data loaders (format={dataset_format})...")
    if dataset_format == "coco":
        train_json = dataset_path / "train" / "_annotations.coco.json"
        train_images = dataset_path / "train" / "images"
        if not train_json.exists():
            print(f"ERROR: Train annotations not found: {train_json}")
            sys.exit(1)
        train_ds = CocoSamDataset(str(train_json), str(train_images), model)
    else:
        train_ds = ImageMaskDataset(
            str(dataset_path / "train" / "images"),
            str(dataset_path / "train" / "masks"),
            model,
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(f"  Train: {len(train_ds)} samples")

    val_loader = None
    if dataset_format == "coco":
        val_json = dataset_path / "valid" / "_annotations.coco.json"
        val_images = dataset_path / "valid" / "images"
        if val_json.exists():
            val_ds = CocoSamDataset(str(val_json), str(val_images), model)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
            )
            print(f"  Val:   {len(val_ds)} samples")
    else:
        # image+mask format: check test/ for validation
        test_images = dataset_path / "test" / "images"
        test_masks = dataset_path / "test" / "masks"
        if test_images.is_dir() and test_masks.is_dir():
            val_ds = ImageMaskDataset(str(test_images), str(test_masks), model)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
            )
            print(f"  Val:   {len(val_ds)} samples")

    # ---- 4. Training ----
    print(f"\n[4/4] Training for {args.epochs} epochs...")

    optimizer = Adam(model.image_encoder.parameters(), lr=args.lr, weight_decay=0)
    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "rank": args.rank,
                "lr": args.lr,
                "hf_dataset": args.hf_dataset,
                "train_samples": len(train_ds),
            },
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_train = mean(train_losses)
        print(f"  Train loss: {avg_train:.4f}")

        log = {"epoch": epoch + 1, "train/loss": avg_train}

        # Validate
        if val_loader is not None:
            val_losses = validate(model, val_loader, loss_fn, device)
            avg_val = mean(val_losses)
            print(f"  Val loss:   {avg_val:.4f}")
            log["val/loss"] = avg_val

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = output_dir / f"lora_rank{args.rank}_best.safetensors"
                sam_lora.save_lora_parameters(str(best_path))
                print(f"  Saved best model -> {best_path}")

        # Visualize predictions on validation set
        if wandb_run and val_loader is not None:
            panels = visualize_predictions(model, val_loader.dataset, device)
            log["val/predictions"] = [
                wandb.Image(p, caption=f"sample {i}: input | GT | pred")
                for i, p in enumerate(panels)
            ]

        if wandb_run:
            wandb_run.log(log)

    # Save final weights
    final_path = output_dir / f"lora_rank{args.rank}.safetensors"
    sam_lora.save_lora_parameters(str(final_path))
    print(f"\nSaved final LoRA weights -> {final_path}")

    if wandb_run:
        wandb_run.finish()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
