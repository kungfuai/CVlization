"""Weights & Biases integration for SAM3 LoRA training.

Provides:
- Config-driven metric logging (reads wandb.metrics from config_lora.yaml)
- Side-by-side val visualizations (input | GT | pred)
"""

import json
import logging
import os
from pathlib import Path

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


def _build_metric_mapper(metric_map):
    """Build a mapper function from the config's wandb.metrics dict.

    The config maps substring keys to clean wandb names, e.g.::

        "train_all_core_loss": "train/loss"
        "coco_eval_bbox_AP_50": "val/AP_50"

    SAM3's trainer emits keys like ``Losses/train_all_core_loss`` (train) and
    ``val_coco/detection/coco_eval_bbox_AP_50`` (val).  We match by checking
    whether any config key is a substring of the trainer key.

    To handle ``val/AP`` (AP without _50/_75), we also emit ``val/AP`` when we
    see ``coco_eval_bbox_AP`` and no more-specific suffix matched.
    """
    # Sort so longer (more specific) keys are checked first.
    sorted_entries = sorted(metric_map.items(), key=lambda kv: -len(kv[0]))
    # Keys that are substrings of the generic AP key — used to avoid false matches.
    ap_specifics = ("AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")

    def mapper(payload):
        out = {}
        for trainer_key, value in payload.items():
            for substr, wandb_name in sorted_entries:
                if substr in trainer_key:
                    out[wandb_name] = value
                    break
            else:
                # Generic AP (not a more-specific variant)
                if "coco_eval_bbox_AP" in trainer_key and not any(
                    s in trainer_key for s in ap_specifics
                ):
                    out["val/box_AP"] = value
                elif "coco_eval_segm_AP" in trainer_key and not any(
                    s in trainer_key for s in ap_specifics
                ):
                    out["val/mask_AP"] = value
            if trainer_key == "Trainer/epoch":
                out["epoch"] = value
        return out

    return mapper


# ── Logger patching ──────────────────────────────────────────────────────────


def patch_trainer_for_wandb(trainer, dataset_dir, output_dir, wandb_cfg):
    """Monkey-patch the trainer to log clean metrics + val images to wandb.

    Args:
        trainer: Instantiated SAM3 Trainer.
        dataset_dir: Path to dataset root (with train/ and test/ subdirs).
        output_dir: Path to experiment output dir.
        wandb_cfg: The ``wandb`` section of config_lora.yaml (OmegaConf node).
    """
    import wandb
    from omegaconf import OmegaConf

    metric_map = OmegaConf.to_container(wandb_cfg.metrics, resolve=True)
    mapper = _build_metric_mapper(metric_map)
    max_val_images = wandb_cfg.get("val_images", 4)

    # --- Patch logger.log_dict to send filtered metrics to wandb ---
    original_log_dict = trainer.logger.log_dict

    def log_dict_with_wandb(payload, step):
        original_log_dict(payload, step)
        mapped = mapper(payload)
        if mapped:
            wandb.log(mapped)

    trainer.logger.log_dict = log_dict_with_wandb

    # --- Patch run_val to log side-by-side images after validation ---
    original_run_val = trainer.run_val

    def run_val_with_images():
        original_run_val()
        _log_val_box_images(
            dataset_dir, output_dir,
            epoch=int(trainer.epoch),
            max_images=max_val_images,
        )
        _log_val_instance_images(
            dataset_dir, output_dir,
            epoch=int(trainer.epoch),
            max_images=max_val_images,
        )

    trainer.run_val = run_val_with_images


# ── Side-by-side visualisation ───────────────────────────────────────────────

MARGIN = 10  # pixels between panels
FONT_SIZE = 14
BBOX_WIDTH = 3
SCORE_THRESHOLD = 0.3  # only show predictions above this confidence


def _draw_bboxes(img, bboxes, color, labels=None):
    """Draw bboxes on a PIL Image. bboxes are [x, y, w, h]."""
    draw = ImageDraw.Draw(img)
    for i, (x, y, w, h) in enumerate(bboxes):
        draw.rectangle([x, y, x + w, y + h], outline=color, width=BBOX_WIDTH)
        if labels and i < len(labels):
            draw.text((x + 2, y + 2), labels[i], fill=color)
    return img


def _make_panel(img, title):
    """Add a title bar above an image."""
    bar_h = 24
    panel = Image.new("RGB", (img.width, img.height + bar_h), (40, 40, 40))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), title, fill=(255, 255, 255))
    panel.paste(img, (0, bar_h))
    return panel


def _compose_side_by_side(panels):
    """Join panels horizontally with a margin."""
    h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + MARGIN * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, h), (200, 200, 200))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + MARGIN
    return canvas


def _load_gt_and_index(dataset_dir):
    """Load GT COCO JSON and index annotations by image_id."""
    gt_file = os.path.join(dataset_dir, "test", "_annotations.coco.json")
    if not os.path.exists(gt_file):
        return None, {}
    with open(gt_file) as f:
        gt = json.load(f)
    gt_by_img = {}
    for ann in gt["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    return gt, gt_by_img


def _log_val_box_images(dataset_dir, output_dir, epoch, max_images=4):
    """Create side-by-side (input | GT boxes | pred boxes) and log to wandb."""
    import wandb

    pred_file = os.path.join(output_dir, "dumps", "coco_predictions_bbox.json")
    gt, gt_by_img = _load_gt_and_index(dataset_dir)
    images_dir = os.path.join(dataset_dir, "test")

    if gt is None or not os.path.exists(pred_file):
        return

    with open(pred_file) as f:
        preds = json.load(f)

    pred_by_img = {}
    for p in preds:
        pred_by_img.setdefault(p["image_id"], []).append(p)

    composites = []
    for img_info in gt["images"][:max_images]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue

        orig = Image.open(img_path).convert("RGB")

        # Panel 1: input
        panel_input = _make_panel(orig.copy(), "Input")

        # Panel 2: GT bboxes (green)
        gt_img = orig.copy()
        gt_bboxes = [a["bbox"] for a in gt_by_img.get(img_id, [])]
        _draw_bboxes(gt_img, gt_bboxes, color=(0, 220, 0))
        panel_gt = _make_panel(gt_img, "Ground Truth")

        # Panel 3: predicted bboxes (cyan, with score)
        pred_img = orig.copy()
        img_preds = pred_by_img.get(img_id, [])
        img_preds = [p for p in img_preds if p["score"] >= SCORE_THRESHOLD]
        img_preds = sorted(img_preds, key=lambda p: p["score"], reverse=True)[:10]
        pred_bboxes = [p["bbox"] for p in img_preds]
        pred_labels = [f"{p['score']:.2f}" for p in img_preds]
        _draw_bboxes(pred_img, pred_bboxes, color=(0, 200, 255), labels=pred_labels)
        panel_pred = _make_panel(pred_img, "Prediction")

        composite = _compose_side_by_side([panel_input, panel_gt, panel_pred])
        caption = f"{fname} — input | gt | pred"
        composites.append(wandb.Image(composite, caption=caption))

    if composites:
        wandb.log({"images/val_box": composites, "epoch": epoch})
        log.info("Logged %d val box images to wandb (epoch %d)", len(composites), epoch)


# ── Instance mask colours ─────────────────────────────────────────────────────

# Distinct colours for up to 10 instances; cycles for more.
INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0),
]


def _overlay_masks(img, masks, alpha=0.45, colors=None, labels=None):
    """Overlay binary masks on a PIL Image with per-instance colours."""
    import numpy as np

    base = np.array(img, dtype=np.float32)
    overlay = base.copy()
    draw = ImageDraw.Draw(img)
    for i, mask in enumerate(masks):
        color = (colors or INSTANCE_COLORS)[i % len(colors or INSTANCE_COLORS)]
        mask_bool = np.asarray(mask, dtype=bool)
        for c in range(3):
            overlay[..., c] = np.where(mask_bool, color[c], overlay[..., c])
    blended = (base * (1 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended)

    # Draw labels at mask centroids
    if labels:
        result_draw = ImageDraw.Draw(result)
        for i, mask in enumerate(masks):
            if i >= len(labels):
                break
            ys, xs = np.where(np.asarray(mask, dtype=bool))
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            result_draw.text((cx, cy), labels[i], fill=(255, 255, 255))
    return result


def _decode_rle_masks(preds, img_h, img_w):
    """Decode RLE-encoded segmentation masks from COCO predictions."""
    import numpy as np

    try:
        from pycocotools import mask as mask_util
    except ImportError:
        log.warning("pycocotools not installed, cannot decode masks")
        return []

    masks = []
    for p in preds:
        seg = p.get("segmentation")
        if seg is None:
            continue
        # Ensure RLE counts is bytes (pycocotools requirement)
        if isinstance(seg.get("counts"), list):
            rle = mask_util.frPyObjects(seg, img_h, img_w)
        else:
            rle = seg
        mask = mask_util.decode(rle)
        masks.append(mask)
    return masks


def _gt_masks_from_coco(gt_anns, gt_coco, img_h, img_w):
    """Decode GT masks from COCO annotations (polygon or RLE)."""
    import numpy as np

    try:
        from pycocotools import mask as mask_util
    except ImportError:
        return []

    masks = []
    for ann in gt_anns:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, list):
            # Polygon format → convert to RLE
            rles = mask_util.frPyObjects(seg, img_h, img_w)
            rle = mask_util.merge(rles)
        elif isinstance(seg, dict):
            # Already RLE
            if isinstance(seg.get("counts"), list):
                rle = mask_util.frPyObjects(seg, img_h, img_w)
            else:
                rle = seg
        else:
            continue
        masks.append(mask_util.decode(rle))
    return masks


def _log_val_instance_images(dataset_dir, output_dir, epoch, max_images=4):
    """Create side-by-side (input | GT masks | pred masks) and log to wandb."""
    import wandb

    pred_file = os.path.join(output_dir, "dumps", "coco_predictions_segm.json")
    gt, gt_by_img = _load_gt_and_index(dataset_dir)
    images_dir = os.path.join(dataset_dir, "test")

    if gt is None or not os.path.exists(pred_file):
        return

    with open(pred_file) as f:
        preds = json.load(f)

    pred_by_img = {}
    for p in preds:
        pred_by_img.setdefault(p["image_id"], []).append(p)

    composites = []
    for img_info in gt["images"][:max_images]:
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            continue

        orig = Image.open(img_path).convert("RGB")
        img_w, img_h = orig.size

        # Panel 1: input
        panel_input = _make_panel(orig.copy(), "Input")

        # Panel 2: GT instance masks (green shades)
        gt_anns = gt_by_img.get(img_id, [])
        gt_masks = _gt_masks_from_coco(gt_anns, gt, img_h, img_w)
        gt_colors = [(0, int(180 + 40 * (i % 2)), 0) for i in range(len(gt_masks))]
        gt_img = _overlay_masks(orig.copy(), gt_masks, colors=gt_colors) if gt_masks else orig.copy()
        panel_gt = _make_panel(gt_img, "GT Masks")

        # Panel 3: predicted instance masks (distinct colours, with score)
        img_preds = pred_by_img.get(img_id, [])
        img_preds = [p for p in img_preds if p["score"] >= SCORE_THRESHOLD]
        img_preds = sorted(img_preds, key=lambda p: p["score"], reverse=True)[:10]
        pred_masks = _decode_rle_masks(img_preds, img_h, img_w)
        pred_labels = [f"{p['score']:.2f}" for p in img_preds[:len(pred_masks)]]
        pred_img = _overlay_masks(orig.copy(), pred_masks, labels=pred_labels) if pred_masks else orig.copy()
        panel_pred = _make_panel(pred_img, "Pred Masks")

        composite = _compose_side_by_side([panel_input, panel_gt, panel_pred])
        caption = f"{fname} — input | gt | pred"
        composites.append(wandb.Image(composite, caption=caption))

    if composites:
        wandb.log({"images/val_instance": composites, "epoch": epoch})
        log.info("Logged %d val instance images to wandb (epoch %d)", len(composites), epoch)
