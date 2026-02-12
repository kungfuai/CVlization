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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rank", type=int, default=64, help="LoRA rank")
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
    @torch.no_grad(), which blocks training. This reimplements the same
    logic with gradients enabled.
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
        outputs.append({"low_res_logits": low_res_masks, "iou_predictions": iou_predictions})
    return outputs


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Run one training epoch. Returns list of per-batch losses."""
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="  train"):
        batch = _batch_to_device(batch, device)
        outputs = sam_forward(model, batch, multimask_output=False)

        # Stack ground truth and predicted masks
        gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
        pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)

        # pred shape: [B, 1, 1, H, W] -> [B, 1, H, W]
        pred = pred.squeeze(1)
        # gt shape: [B, H, W] -> [B, 1, H, W], resized to match pred
        gt = gt.unsqueeze(1).float().to(device)
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")

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
        outputs = model(batched_input=batch, multimask_output=False)

        gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
        pred = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        pred = pred.squeeze(1)
        gt = gt.unsqueeze(1).float().to(device)
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")

        loss = loss_fn(pred, gt)
        losses.append(loss.item())
    return losses


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
