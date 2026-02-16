import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import mmdet
import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmdet.registry import HOOKS
from mmdet.utils import register_all_modules

from cvlization.dataset.coco_panoptic_tiny import CocoPanopticTinyDatasetBuilder


def get_cache_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_config_path(config_path: str) -> str:
    if config_path.startswith("mmdet::"):
        rel = config_path.split("::", 1)[1]
        root = Path(mmdet.__file__).resolve().parent
        return str(root / ".mim" / "configs" / rel)
    return config_path


def _set_num_classes(cfg: Config, num_things_classes: int, num_stuff_classes: int) -> None:
    if hasattr(cfg.model, "roi_head"):
        bbox_head = cfg.model.roi_head.bbox_head
        if isinstance(bbox_head, list):
            for head in bbox_head:
                head.num_classes = num_things_classes
        else:
            bbox_head.num_classes = num_things_classes

        mask_head = cfg.model.roi_head.mask_head
        if isinstance(mask_head, list):
            for head in mask_head:
                head.num_classes = num_things_classes
        else:
            mask_head.num_classes = num_things_classes

    if hasattr(cfg.model, "semantic_head"):
        cfg.model.semantic_head.num_things_classes = num_things_classes
        cfg.model.semantic_head.num_stuff_classes = num_stuff_classes

    if hasattr(cfg.model, "panoptic_head"):
        cfg.model.panoptic_head.num_things_classes = num_things_classes
        cfg.model.panoptic_head.num_stuff_classes = num_stuff_classes

    if hasattr(cfg.model, "panoptic_fusion_head"):
        cfg.model.panoptic_fusion_head.num_things_classes = num_things_classes
        cfg.model.panoptic_fusion_head.num_stuff_classes = num_stuff_classes


def _set_dataset_cfg(cfg: Config, dsb: CocoPanopticTinyDatasetBuilder) -> None:
    data_root = os.path.join(dsb.data_dir, dsb.dataset_folder)
    img_prefix = f"{dsb.img_folder}/"
    seg_prefix = f"{dsb.seg_folder}/"
    thing_classes = tuple([c["name"] for c in dsb.things_classes])
    stuff_classes = tuple([c["name"] for c in dsb.stuff_classes])
    classes = tuple(dsb.classes)
    metainfo = dict(
        classes=classes,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
    )

    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file = dsb.train_ann_file
    cfg.train_dataloader.dataset.data_prefix = dict(img=img_prefix, seg=seg_prefix)
    cfg.train_dataloader.dataset.metainfo = metainfo

    cfg.val_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.ann_file = dsb.val_ann_file
    cfg.val_dataloader.dataset.data_prefix = dict(img=img_prefix, seg=seg_prefix)
    cfg.val_dataloader.dataset.metainfo = metainfo

    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = dsb.val_ann_file
    cfg.test_dataloader.dataset.data_prefix = dict(img=img_prefix, seg=seg_prefix)
    cfg.test_dataloader.dataset.metainfo = metainfo

    cfg.val_evaluator.ann_file = os.path.join(data_root, dsb.val_ann_file)
    cfg.val_evaluator.seg_prefix = os.path.join(data_root, dsb.seg_folder)
    cfg.test_evaluator = cfg.val_evaluator


def _set_tracking(cfg: Config, enabled: bool, run_name: str) -> None:
    if not enabled:
        return
    cfg.visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=[
            dict(type="LocalVisBackend"),
            dict(type="WandbVisBackend", init_kwargs=dict(project="cvlab", name=run_name)),
        ],
        name="visualizer",
    )


@HOOKS.register_module()
class InputGtPredVisualizationHook(Hook):
    """Log side-by-side panels to visual backends: input | gt | pred."""

    def __init__(
        self,
        draw: bool = True,
        every_n_epochs: int = 1,
        max_samples: int = 2,
        score_thr: float = 0.3,
        backend_args: dict | None = None,
    ):
        self.draw = draw
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_samples = max(1, int(max_samples))
        self.score_thr = score_thr
        self.backend_args = backend_args
        self._visualizer: Visualizer = Visualizer.get_current_instance()

    def _should_draw_this_epoch(self, runner: Runner) -> bool:
        return ((runner.epoch + 1) % self.every_n_epochs) == 0

    @staticmethod
    def _attach_gt(pred_sample, gt_sample):
        for k in ("gt_instances", "gt_sem_seg", "gt_panoptic_seg"):
            if hasattr(gt_sample, k):
                setattr(pred_sample, k, getattr(gt_sample, k))
        if hasattr(pred_sample, "gt_instances") and hasattr(pred_sample.gt_instances, "bboxes"):
            bboxes = pred_sample.gt_instances.bboxes
            if hasattr(bboxes, "tensor"):
                pred_sample.gt_instances.bboxes = bboxes.tensor
        return pred_sample

    @staticmethod
    def _load_rgb(img_path: str, backend_args: dict | None = None) -> np.ndarray:
        img_bytes = get(img_path, backend_args=backend_args)
        return mmcv.imfrombytes(img_bytes, channel_order="rgb")

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs) -> None:
        if not self.draw:
            return
        if not self._should_draw_this_epoch(runner):
            return
        if batch_idx >= self.max_samples:
            return
        if not outputs:
            return

        pred_sample = outputs[0].cpu()
        gt_sample = data_batch["data_samples"][0].cpu() if "data_samples" in data_batch else None
        if gt_sample is not None:
            pred_sample = self._attach_gt(pred_sample, gt_sample)

        img = self._load_rgb(pred_sample.img_path, backend_args=self.backend_args)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            stitched_path = f.name
        try:
            try:
                self._visualizer.add_datasample(
                    "val_gt_pred_tmp",
                    img,
                    data_sample=pred_sample,
                    draw_gt=True,
                    draw_pred=True,
                    show=False,
                    out_file=stitched_path,
                    pred_score_thr=self.score_thr,
                    step=runner.iter + batch_idx,
                )
            except ValueError:
                # Fallback for rare empty panoptic predictions that crash mmdet drawer.
                if hasattr(pred_sample, "pred_panoptic_seg"):
                    delattr(pred_sample, "pred_panoptic_seg")
                if hasattr(pred_sample, "gt_panoptic_seg"):
                    delattr(pred_sample, "gt_panoptic_seg")
                self._visualizer.add_datasample(
                    "val_gt_pred_tmp",
                    img,
                    data_sample=pred_sample,
                    draw_gt=True,
                    draw_pred=True,
                    show=False,
                    out_file=stitched_path,
                    pred_score_thr=self.score_thr,
                    step=runner.iter + batch_idx,
                )
            gt_pred = mmcv.imread(stitched_path, channel_order="rgb")
        finally:
            if os.path.exists(stitched_path):
                os.remove(stitched_path)

        margin = np.full((img.shape[0], 4, 3), 255, dtype=np.uint8)
        panel = np.concatenate([img, margin, gt_pred], axis=1)
        self._visualizer.add_image(
            f"val/input_gt_pred_{batch_idx}",
            panel,
            step=runner.epoch + 1,
        )


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        cache_dir = str(get_cache_dir())
        print(f"Using cache directory: {cache_dir}")
        dsb = CocoPanopticTinyDatasetBuilder(preload=True, flavor=None, data_dir=cache_dir)

        register_all_modules(init_default_scope=False)
        resolved_config = _resolve_config_path(self.args.config)
        cfg = Config.fromfile(resolved_config)

        _set_dataset_cfg(cfg, dsb)
        _set_num_classes(
            cfg,
            num_things_classes=len(dsb.things_classes),
            num_stuff_classes=len(dsb.stuff_classes),
        )
        _set_tracking(
            cfg,
            enabled=self.args.track and bool(os.environ.get("WANDB_API_KEY")),
            run_name=self.args.run_name or "panoptic-mmdet3",
        )
        if self.args.track and bool(os.environ.get("WANDB_API_KEY")):
            cfg.custom_hooks = cfg.get("custom_hooks", [])
            cfg.custom_hooks.append(
                dict(
                    type="InputGtPredVisualizationHook",
                    draw=True,
                    every_n_epochs=self.args.log_images_every_n_epochs,
                    max_samples=self.args.max_logged_images_per_epoch,
                    score_thr=self.args.log_images_score_thr,
                )
            )

        cfg.train_dataloader.batch_size = self.args.batch_size
        cfg.train_dataloader.num_workers = self.args.num_workers
        cfg.val_dataloader.num_workers = self.args.num_workers
        cfg.test_dataloader.num_workers = self.args.num_workers

        cfg.optim_wrapper.optimizer.lr = self.args.lr
        cfg.train_cfg.max_epochs = self.args.max_epochs
        cfg.val_interval = self.args.val_interval
        cfg.default_hooks.logger.interval = self.args.log_interval
        cfg.default_hooks.checkpoint.interval = self.args.checkpoint_interval
        cfg.work_dir = self.args.work_dir
        cfg.load_from = self.args.load_from
        cfg.launcher = "none"
        cfg.resume = False

        print(
            "train config:",
            dict(
                config=self.args.config,
                resolved_config=resolved_config,
                batch_size=cfg.train_dataloader.batch_size,
                lr=cfg.optim_wrapper.optimizer.lr,
                max_epochs=cfg.train_cfg.max_epochs,
                work_dir=cfg.work_dir,
            ),
        )
        runner = Runner.from_cfg(cfg)
        runner.train()


if __name__ == "__main__":
    parser = ArgumentParser(
        epilog="MMDetection 3.x panoptic training on COCO Panoptic Tiny."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="mmdet::panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py",
        help="MMDetection config path.",
    )
    parser.add_argument("--work-dir", type=str, default="./tmp")
    parser.add_argument(
        "--load-from",
        type=str,
        default="https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth",
        help="Pretrained checkpoint URL or local path (set to empty string to train from scratch).",
    )
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-images-every-n-epochs", type=int, default=1)
    parser.add_argument("--max-logged-images-per-epoch", type=int, default=2)
    parser.add_argument("--log-images-score-thr", type=float, default=0.3)
    args = parser.parse_args()
    TrainingSession(args).run()
