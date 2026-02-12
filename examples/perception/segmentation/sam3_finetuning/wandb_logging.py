"""Weights & Biases logging integration for SAM3 fine-tuning."""
import json
from pathlib import Path

import numpy as np

from dataset import SUPERCATEGORY


def _log_val_predictions_to_wandb(
    predictions_json: str,
    gt_json: str,
    images_dir: str,
    max_images: int = 8,
    epoch: int | None = None,
):
    """Log side-by-side GT vs prediction images to wandb.

    Each logged image is a horizontal pair: left = ground truth (masks + bboxes),
    right = model predictions (bboxes, and segmentation masks if available).
    """
    import wandb
    from PIL import Image, ImageDraw

    if not Path(predictions_json).exists() or not Path(gt_json).exists():
        return

    with open(predictions_json) as f:
        predictions = json.load(f)
    with open(gt_json) as f:
        gt_data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in gt_data["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in gt_data.get("categories", [])}
    gt_by_image: dict[int, list] = {}
    for ann in gt_data.get("annotations", []):
        gt_by_image.setdefault(ann["image_id"], []).append(ann)
    pred_by_image: dict[int, list] = {}
    for pred in predictions:
        pred_by_image.setdefault(pred["image_id"], []).append(pred)

    # Color palette (RGBA with alpha for semi-transparent mask overlay)
    MASK_COLORS = [
        (0, 200, 0, 90), (200, 120, 0, 90), (0, 100, 255, 90),
        (255, 0, 200, 90), (200, 200, 0, 90), (0, 200, 200, 90),
    ]

    sample_ids = sorted(
        pred_by_image.keys(),
        key=lambda x: len(pred_by_image[x]),
        reverse=True,
    )[:max_images]
    if not sample_ids:
        sample_ids = list(id_to_file.keys())[:max_images]

    wandb_images = []
    for img_id in sample_ids:
        fname = id_to_file.get(img_id)
        if not fname:
            continue
        img_path = Path(images_dir) / fname
        if not img_path.exists():
            continue

        base_img = Image.open(img_path).convert("RGBA")

        # --- Left panel: Ground Truth (masks + bboxes) ---
        gt_img = base_img.copy()
        overlay = Image.new("RGBA", gt_img.size, (0, 0, 0, 0))
        for ann in gt_by_image.get(img_id, []):
            seg = ann.get("segmentation")
            if seg and isinstance(seg, dict) and "counts" in seg:
                try:
                    from pycocotools import mask as mask_utils
                    rle = dict(seg)
                    if isinstance(rle["counts"], str):
                        rle["counts"] = rle["counts"].encode("utf-8")
                    mask_arr = mask_utils.decode(rle)
                    cat_id = ann.get("category_id", 0)
                    color = MASK_COLORS[cat_id % len(MASK_COLORS)]
                    colored = Image.new("RGBA", gt_img.size, color)
                    mask_bool = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    overlay = Image.composite(colored, overlay, mask_bool)
                except Exception:
                    pass
        gt_img = Image.alpha_composite(gt_img, overlay).convert("RGB")
        gt_draw = ImageDraw.Draw(gt_img)
        for ann in gt_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cat_name = cat_id_to_name.get(ann.get("category_id"), "?")
            gt_draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
            gt_draw.text((x, max(0, y - 12)), cat_name, fill="lime")
        gt_draw.text((4, 4), "Ground Truth", fill="white")

        # --- Right panel: Predictions (bboxes + optional masks) ---
        pred_img = base_img.copy().convert("RGB")
        pred_draw = ImageDraw.Draw(pred_img)
        for pred in pred_by_image.get(img_id, []):
            score = pred.get("score", 0)
            if score < 0.3:
                continue
            x, y, w, h = pred["bbox"]
            cat_name = cat_id_to_name.get(pred.get("category_id"), "?")
            pred_draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            pred_draw.text((x, max(0, y - 12)), f"{cat_name} {score:.2f}", fill="red")
        pred_draw.text((4, 4), "Predictions", fill="white")

        # --- Stitch side by side ---
        w_img, h_img = gt_img.size
        combined = Image.new("RGB", (w_img * 2 + 4, h_img), (40, 40, 40))
        combined.paste(gt_img, (0, 0))
        combined.paste(pred_img, (w_img + 4, 0))

        epoch_str = f"epoch {epoch} | " if epoch is not None else ""
        wandb_images.append(wandb.Image(combined, caption=f"{epoch_str}{fname}"))

    if wandb_images:
        wandb.log({"val/predictions": wandb_images})


def setup_wandb_logging(project: str, run_name: str | None, config: dict, output_dir: str, dataset_dir: str):
    """Initialize wandb and install a WandbLogger subclass into SAM3's trainer."""
    import wandb
    from sam3.train.utils.logger import Logger
    import sam3.train.trainer as _trainer_mod

    wandb.init(project=project, name=run_name, config=config)

    sc = SUPERCATEGORY
    pred_json = str(Path(output_dir) / "dumps" / sc / "coco_predictions_bbox.json")
    gt_json = str(Path(dataset_dir) / sc / "test" / "_annotations.coco.json")
    images_dir = str(Path(dataset_dir) / sc / "test")

    def _clean_key(k: str) -> str:
        return k.replace("Meters_train/", "")

    _WANDB_EXACT_SUFFIXES = {
        "train_all_loss",
        "train_all_loss_bbox",
        "train_all_loss_giou",
        "train_all_loss_ce",
        "train_all_ce_f1",
    }

    def _keep(k: str) -> bool:
        base = k.rsplit("/", 1)[-1] if "/" in k else k
        if base in _WANDB_EXACT_SUFFIXES:
            return True
        if "coco_eval" in k:
            return True
        if k.startswith("Optim/") and "/lr" in k:
            return True
        return False

    class WandbLogger(Logger):
        """Extends SAM3's Logger to also forward metrics to W&B."""

        def log_dict(self, payload, step):
            super().log_dict(payload, step)
            cleaned = {_clean_key(k): v for k, v in payload.items() if _keep(k)}
            if cleaned:
                wandb.log(cleaned)
            if any("coco_eval" in k for k in payload):
                try:
                    _log_val_predictions_to_wandb(
                        pred_json, gt_json, images_dir, epoch=step,
                    )
                except Exception as e:
                    print(f"Warning: failed to log val predictions to wandb: {e}")

        def log(self, name, data, step):
            super().log(name, data, step)
            if _keep(name):
                wandb.log({_clean_key(name): data, "trainer_step": step})

    _trainer_mod.Logger = WandbLogger
    print("✓ W&B logging enabled (scalars + side-by-side GT/pred images)")
