"""Train / finetune Detectron2 PanopticFPN on coco_panoptic_tiny."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.utils.logger import setup_logger
from panopticapi.utils import rgb2id
from pycocotools import mask as mask_util

from cvlization.dataset.coco_panoptic_tiny import CocoPanopticTinyDatasetBuilder


# ---------------------------------------------------------------------------
# Dataset conversion helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization" / "data"


def convert_panoptic_to_instances(panoptic_json_path: str, seg_dir: str, out_json_path: str) -> None:
    """Extract thing segments from panoptic PNGs and write COCO instances JSON with RLE masks."""
    with open(panoptic_json_path) as f:
        pan = json.load(f)

    things_cat_ids = {c["id"] for c in pan["categories"] if c["isthing"] == 1}

    images = []
    annotations = []
    ann_id = 1

    for img_info in pan["images"]:
        images.append(img_info)

    img_id_to_info = {img["id"]: img for img in pan["images"]}

    for ann_entry in pan["annotations"]:
        image_id = ann_entry["image_id"]
        seg_filename = ann_entry["file_name"]
        seg_path = os.path.join(seg_dir, seg_filename)
        if not os.path.exists(seg_path):
            continue

        pan_png = np.array(cv2.imread(seg_path, cv2.IMREAD_COLOR)[:, :, ::-1])  # BGR->RGB
        pan_ids = rgb2id(pan_png)
        img_info = img_id_to_info[image_id]
        h, w = img_info["height"], img_info["width"]

        for seg in ann_entry["segments_info"]:
            if seg["category_id"] not in things_cat_ids:
                continue
            if seg.get("iscrowd", 0):
                continue

            binary_mask = (pan_ids == seg["id"]).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            bbox = mask_util.toBbox(rle).tolist()  # [x, y, w, h]
            area = float(mask_util.area(rle))

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": seg["category_id"],
                "segmentation": rle,
                "bbox": bbox,
                "bbox_mode": 0,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    # Only keep thing categories
    categories = [c for c in pan["categories"] if c["isthing"] == 1]

    out = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(annotations)} instance annotations to {out_json_path}")


def create_semantic_masks(panoptic_json_path: str, seg_dir: str, sem_out_dir: str,
                          stuff_dataset_id_to_contiguous_id: dict) -> None:
    """Create grayscale semantic segmentation PNGs from panoptic PNGs.

    PanopticFPN semantic head only predicts stuff classes.
    Stuff pixels -> contiguous stuff ID (0..N-1), thing pixels -> 255 (ignore).
    """
    with open(panoptic_json_path) as f:
        pan = json.load(f)

    stuff_cat_ids = set(stuff_dataset_id_to_contiguous_id.keys())
    os.makedirs(sem_out_dir, exist_ok=True)

    for ann_entry in pan["annotations"]:
        seg_filename = ann_entry["file_name"]
        seg_path = os.path.join(seg_dir, seg_filename)
        if not os.path.exists(seg_path):
            continue

        pan_png = np.array(cv2.imread(seg_path, cv2.IMREAD_COLOR)[:, :, ::-1])
        pan_ids = rgb2id(pan_png)

        sem_map = np.full(pan_ids.shape, 255, dtype=np.uint8)
        for seg in ann_entry["segments_info"]:
            cat_id = seg["category_id"]
            if cat_id in stuff_cat_ids:
                sem_map[pan_ids == seg["id"]] = stuff_dataset_id_to_contiguous_id[cat_id]

        out_path = os.path.join(sem_out_dir, seg_filename)
        cv2.imwrite(out_path, sem_map)

    print(f"Wrote semantic masks to {sem_out_dir}")


def prepare_separated_dataset(data_root: str, ann_json: str, seg_dir: str, split_name: str) -> dict:
    """Prepare instance JSON + semantic PNGs from panoptic annotations.

    Returns dict with paths to the generated files.
    """
    out_dir = os.path.join(data_root, f"detectron2_{split_name}")
    inst_json = os.path.join(out_dir, "instances.json")
    sem_dir = os.path.join(out_dir, "semantic_masks")

    if not os.path.exists(inst_json):
        convert_panoptic_to_instances(ann_json, seg_dir, inst_json)
    else:
        print(f"Instance JSON already exists: {inst_json}")

    return {
        "instances_json": inst_json,
        "sem_seg_dir": sem_dir,
        "panoptic_json": ann_json,
        "panoptic_dir": seg_dir,
    }


def register_dataset(dsb: CocoPanopticTinyDatasetBuilder, split: str) -> str:
    """Register a COCO panoptic dataset split with detectron2.

    Uses register_coco_panoptic_separated for full PanopticFPN training.
    Returns the registered dataset name.
    """
    from detectron2.data.datasets import register_coco_panoptic_separated

    data_root = os.path.join(dsb.data_dir, dsb.dataset_folder)
    img_dir = os.path.join(data_root, dsb.img_folder if hasattr(dsb, 'img_folder') else "val2017_subset")
    seg_dir = os.path.join(data_root, dsb.seg_folder if hasattr(dsb, 'seg_folder') else "val2017_subset_panoptic_masks")

    if split == "train":
        ann_json = os.path.join(data_root, dsb.train_ann_file)
    else:
        ann_json = os.path.join(data_root, dsb.val_ann_file)

    # Load annotation to get categories
    with open(ann_json) as f:
        ann_data = json.load(f)

    all_categories = ann_data["categories"]
    thing_classes = [c["name"] for c in all_categories if c["isthing"] == 1]
    stuff_classes = [c["name"] for c in all_categories if c["isthing"] == 0]
    thing_ids = [c["id"] for c in all_categories if c["isthing"] == 1]
    stuff_ids = [c["id"] for c in all_categories if c["isthing"] == 0]

    # Build contiguous ID mappings (detectron2 convention)
    thing_dataset_id_to_contiguous_id = {did: i for i, did in enumerate(thing_ids)}
    stuff_dataset_id_to_contiguous_id = {did: i for i, did in enumerate(stuff_ids)}

    # Prepare separated data (instances JSON + semantic masks)
    separated = prepare_separated_dataset(data_root, ann_json, seg_dir, split)

    # Build semantic masks: stuff-only contiguous IDs, things=255 (ignore)
    sem_dir = separated["sem_seg_dir"]
    if not os.path.exists(sem_dir) or len(os.listdir(sem_dir)) == 0:
        create_semantic_masks(ann_json, seg_dir, sem_dir, stuff_dataset_id_to_contiguous_id)

    dataset_name = f"coco_panoptic_tiny_{split}"

    # Clean up any previous registration
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    for suffix in ["_separated", "_stuffonly", "_separated_stuffonly"]:
        name = dataset_name + suffix
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)

    register_coco_panoptic_separated(
        name=dataset_name,
        metadata={},
        image_root=img_dir,
        panoptic_root=seg_dir,
        panoptic_json=ann_json,
        sem_seg_root=sem_dir,
        instances_json=separated["instances_json"],
    )

    # Set metadata for the registered datasets (always set on all variants)
    for suffix in ["", "_separated", "_stuffonly"]:
        name = dataset_name + suffix
        meta = MetadataCatalog.get(name)
        meta.set(
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            panoptic_json=ann_json,
            panoptic_root=seg_dir,
            image_root=img_dir,
        )

    return dataset_name


# ---------------------------------------------------------------------------
# Wandb hooks
# ---------------------------------------------------------------------------

class WandbScalarHook(HookBase):
    """Forward detectron2 EventStorage scalars to wandb every N iterations."""

    def __init__(self, log_period: int = 20):
        self.log_period = log_period

    def after_step(self):
        import wandb
        if wandb.run is None:
            return
        iteration = self.trainer.iter
        if (iteration + 1) % self.log_period != 0:
            return
        storage = self.trainer.storage
        log_dict = {}
        for k, (v, _iter) in storage.latest().items():
            if isinstance(v, (int, float)):
                log_dict[k] = v
        if log_dict:
            wandb.log(log_dict, step=iteration)


class WandbPanopticImageHook(HookBase):
    """Log side-by-side [input | gt | pred] panels for semantic + instance to wandb."""

    MARGIN = 4

    def __init__(self, cfg, eval_period: int, num_images: int = 3):
        self.cfg = cfg.clone()
        self.eval_period = eval_period
        self.num_images = num_images
        self._cmap = self._build_colormap(256)
        # Pre-load val samples and category mapping once
        self._samples = None
        self._cat_id_to_idx = None

    @staticmethod
    def _build_colormap(n):
        rng = np.random.RandomState(42)
        cmap = rng.randint(60, 220, size=(n, 3), dtype=np.uint8)
        cmap[255] = 0
        return cmap

    def _colorize(self, label_map):
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cid in np.unique(label_map):
            rgb[label_map == cid] = self._cmap[int(cid) % len(self._cmap)]
        return rgb

    def _hmargin(self, h):
        return np.full((h, self.MARGIN, 3), 255, dtype=np.uint8)

    def _ensure_val_samples(self):
        """Load val samples and category mapping once, cache for reuse."""
        if self._samples is not None:
            return
        val_name = self.cfg.DATASETS.TEST[0]
        base_name = val_name.replace("_separated", "")
        meta = MetadataCatalog.get(base_name)

        with open(meta.panoptic_json) as f:
            pan = json.load(f)

        all_cats = pan["categories"]
        self._cat_id_to_idx = {c["id"]: i for i, c in enumerate(all_cats)}

        # Build inverse maps: model contiguous ID → original dataset category ID.
        # Detectron2 predictions use contiguous IDs (things 0..79, stuff 0..52)
        # but GT uses original COCO IDs. We need to convert pred IDs back.
        self._contiguous_thing_to_dataset = {
            v: k for k, v in meta.thing_dataset_id_to_contiguous_id.items()
        }
        self._contiguous_stuff_to_dataset = {
            v: k for k, v in meta.stuff_dataset_id_to_contiguous_id.items()
        }

        self._samples = []
        img_id_to_info = {img["id"]: img for img in pan["images"]}
        for ann_entry in pan["annotations"][: self.num_images]:
            img_info = img_id_to_info[ann_entry["image_id"]]
            img_path = os.path.join(meta.image_root, img_info["file_name"])
            seg_path = os.path.join(meta.panoptic_root, ann_entry["file_name"])
            sid2idx = {
                seg["id"]: self._cat_id_to_idx[seg["category_id"]]
                for seg in ann_entry["segments_info"]
                if seg["category_id"] in self._cat_id_to_idx
            }
            self._samples.append((img_path, seg_path, sid2idx))

    def after_step(self):
        iteration = self.trainer.iter + 1
        if iteration % self.eval_period != 0 and iteration != self.trainer.max_iter:
            return

        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        self._ensure_val_samples()

        # Build predictor without loading pretrained weights (we use trainer weights)
        pred_cfg = self.cfg.clone()
        pred_cfg.MODEL.WEIGHTS = ""
        predictor = DefaultPredictor(pred_cfg)
        predictor.model.load_state_dict(self.trainer.model.state_dict())

        sem_panels, inst_panels = [], []

        for img_path, seg_path, sid2idx in self._samples:
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            h, w = raw_img.shape[:2]
            margin = self._hmargin(h)

            # GT panoptic
            pan_png = cv2.imread(seg_path, cv2.IMREAD_COLOR)
            pan_png = cv2.cvtColor(pan_png, cv2.COLOR_BGR2RGB)
            seg_ids = rgb2id(pan_png)

            gt_sem = np.full_like(seg_ids, 255, dtype=np.int32)
            for sid, cls_idx in sid2idx.items():
                gt_sem[seg_ids == sid] = cls_idx

            # Prediction
            outputs = predictor(cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            panoptic_seg = panoptic_seg.cpu().numpy()

            # Pred semantic: convert model contiguous IDs back to dataset IDs,
            # then map to the same unified index used for GT colorization.
            pred_sem = np.full_like(panoptic_seg, 255, dtype=np.int32)
            for s in segments_info:
                contiguous_id = s["category_id"]
                if s["isthing"]:
                    dataset_id = self._contiguous_thing_to_dataset.get(contiguous_id)
                else:
                    dataset_id = self._contiguous_stuff_to_dataset.get(contiguous_id)
                if dataset_id is not None and dataset_id in self._cat_id_to_idx:
                    pred_sem[panoptic_seg == s["id"]] = self._cat_id_to_idx[dataset_id]

            # Semantic panel: [input | gt_semantic | pred_semantic]
            gt_sem_rgb = self._colorize(gt_sem)
            pred_sem_rgb = self._colorize(pred_sem)
            sem_panels.append(np.concatenate(
                [raw_img, margin, gt_sem_rgb, margin, pred_sem_rgb], axis=1
            ))

            # Instance panel: [input | gt_instance | pred_instance]
            gt_inst_rgb = self._colorize(seg_ids)
            pred_inst_rgb = self._colorize(panoptic_seg)
            inst_panels.append(np.concatenate(
                [raw_img, margin, gt_inst_rgb, margin, pred_inst_rgb], axis=1
            ))

        if sem_panels:
            wandb.log({
                "images/val_semantic": [
                    wandb.Image(p, caption="input | gt | pred") for p in sem_panels
                ],
                "images/val_instance": [
                    wandb.Image(p, caption="input | gt | pred") for p in inst_panels
                ],
            }, step=self.trainer.iter)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PanopticTrainer(DefaultTrainer):
    def run_step(self):
        """Override to handle NaN sem_seg loss gracefully.

        Some batches contain only "thing" pixels (sem_seg mask all 255/ignore),
        which makes cross_entropy return NaN.  Since detectron2 raises
        FloatingPointError *before* backward() is called, the model weights are
        not corrupted and we can safely skip the step.
        """
        try:
            self._trainer.run_step()
        except FloatingPointError:
            import logging
            logging.getLogger("detectron2").warning(
                f"NaN/Inf loss at iter {self.iter} – skipping step (likely all-ignore sem_seg batch)"
            )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        # COCOPanopticEvaluator needs the base panoptic dataset name
        base_name = dataset_name.replace("_separated", "")
        return COCOPanopticEvaluator(base_name, output_folder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_cfg(args, train_dataset_name: str, val_dataset_name: str, num_thing_classes: int, num_stuff_classes: int):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )

    cfg.DATASETS.TRAIN = (f"{train_dataset_name}_separated",)
    cfg.DATASETS.TEST = (f"{val_dataset_name}_separated",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.WARMUP_ITERS = min(500, args.max_iter // 5)
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))
    cfg.SOLVER.CHECKPOINT_PERIOD = max(args.eval_period, 500)

    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_thing_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_stuff_classes

    cfg.INPUT.MASK_FORMAT = "bitmask"

    # Gradient clipping to prevent NaN in sem_seg head (randomly initialized)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.OUTPUT_DIR = os.path.join("output", "panoptic_detectron2")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Train Detectron2 PanopticFPN on coco_panoptic_tiny")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-period", type=int, default=500)
    parser.add_argument("--freeze-at", type=int, default=2)
    parser.add_argument("--track", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Download / prepare dataset
    cache_dir = str(_get_cache_dir())
    print(f"Using cache directory: {cache_dir}")
    dsb = CocoPanopticTinyDatasetBuilder(preload=True, flavor=None, data_dir=cache_dir)

    # Register train and val splits with detectron2
    train_name = register_dataset(dsb, "train")
    val_name = register_dataset(dsb, "val")

    num_thing_classes = len(dsb.things_classes)
    num_stuff_classes = len(dsb.stuff_classes)
    print(f"Things: {num_thing_classes}, Stuff: {num_stuff_classes}")

    cfg = build_cfg(args, train_name, val_name, num_thing_classes, num_stuff_classes)

    # Wandb init
    if args.track:
        try:
            import wandb
            wandb.init(
                project="panoptic-detectron2",
                config={
                    "max_iter": args.max_iter,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "freeze_at": args.freeze_at,
                    "model": "panoptic_fpn_R_50_3x",
                },
            )
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")

    trainer = PanopticTrainer(cfg)

    # Register wandb hooks (scalars + images)
    if args.track:
        try:
            import wandb
            if wandb.run is not None:
                scalar_hook = WandbScalarHook(log_period=20)
                image_hook = WandbPanopticImageHook(cfg, eval_period=args.eval_period)
                trainer.register_hooks([scalar_hook, image_hook])
                # Move image hook to end (after eval)
                trainer._hooks = trainer._hooks[:-1] + trainer._hooks[-1:]
        except Exception as e:
            print(f"Warning: wandb hook registration failed: {e}")

    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    if args.track:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
