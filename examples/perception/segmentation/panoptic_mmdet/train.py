import os
from pathlib import Path
import numpy as np
try:
    from mmdet.datasets import build_dataset
except ImportError:
    print("mmdet not installed")
    print("For torch 1.11.*:")
    print("pip install mmdet==2.24.1")
    print(
        "pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html"
    )
    print("For torch 1.12.*:")
    print("pip install mmdet==2.25.1")
    print(
        "pip install -U mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12.0/index.html"
    )
    raise
import mmcv
from mmcv.runner import HOOKS, Hook
from mmdet.core import INSTANCE_OFFSET
from cvlization.dataset.coco_panoptic_tiny import CocoPanopticTinyDatasetBuilder
from cvlization.torch.net.panoptic_segmentation.mmdet import (
    MMPanopticSegmentationModels,
    MMDatasetAdaptor,
    MMTrainer,
)


@HOOKS.register_module()
class WandbPanopticImageHook(Hook):
    """Log side-by-side (input | gt_semantic | pred_semantic) to wandb."""

    MARGIN = 4

    def __init__(self, img_paths, seg_paths, seg_id_to_class_idx_maps,
                 num_images=3, log_every_n_epochs=1):
        self.img_paths = img_paths[:num_images]
        self.seg_paths = seg_paths[:num_images]
        self.seg_id_to_class_idx_maps = seg_id_to_class_idx_maps[:num_images]
        self.log_every_n_epochs = log_every_n_epochs
        self._cmap = self._build_colormap(256)

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

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.log_every_n_epochs != 0:
            return
        import wandb
        if wandb.run is None:
            return

        from mmdet.apis import inference_detector
        from panopticapi.utils import rgb2id

        model = runner.model
        if hasattr(model, "module"):
            model = model.module  # unwrap DataParallel

        sem_panels, inst_panels = [], []
        for img_path, seg_path, sid2idx in zip(
            self.img_paths, self.seg_paths, self.seg_id_to_class_idx_maps
        ):
            raw_img = mmcv.bgr2rgb(mmcv.imread(img_path))
            h, w = raw_img.shape[:2]
            margin = self._hmargin(h)

            # GT panoptic PNG
            pan_png = mmcv.bgr2rgb(mmcv.imread(seg_path, flag="color"))
            seg_ids = rgb2id(pan_png)

            # GT semantic: segment_id → contiguous class index
            gt_sem = np.full_like(seg_ids, 255, dtype=np.int32)
            for sid, cls_idx in sid2idx.items():
                gt_sem[seg_ids == sid] = cls_idx

            # Prediction
            result = inference_detector(model, img_path)
            pan = result.get("pan_results", None)
            if pan is not None:
                pred_sem = (pan % INSTANCE_OFFSET).astype(np.int32)
            else:
                pan = np.zeros(raw_img.shape[:2], dtype=np.int32)
                pred_sem = pan

            # Semantic panel (class-level colors)
            gt_sem_rgb = mmcv.imresize(self._colorize(gt_sem), (w, h))
            pred_sem_rgb = mmcv.imresize(self._colorize(pred_sem), (w, h))
            sem_panels.append(np.concatenate(
                [raw_img, margin, gt_sem_rgb, margin, pred_sem_rgb], axis=1,
            ))

            # Instance panel (unique color per instance)
            gt_inst_rgb = mmcv.imresize(self._colorize(seg_ids), (w, h))
            pred_inst_rgb = mmcv.imresize(self._colorize(pan), (w, h))
            inst_panels.append(np.concatenate(
                [raw_img, margin, gt_inst_rgb, margin, pred_inst_rgb], axis=1,
            ))

        wandb.log(
            {
                "images/val_semantic": [
                    wandb.Image(p, caption="input | gt | pred")
                    for p in sem_panels
                ],
                "images/val_instance": [
                    wandb.Image(p, caption="input | gt | pred")
                    for p in inst_panels
                ],
            },
            step=runner.iter,
        )


def get_cache_dir() -> Path:
    """Get the CVlization cache directory for datasets."""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "cvlization" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        # Use centralized cache directory for datasets
        cache_dir = str(get_cache_dir())
        print(f"Using cache directory: {cache_dir}")
        dsb = CocoPanopticTinyDatasetBuilder(preload=True, flavor=None, data_dir=cache_dir)
        self.model, self.cfg = self.create_model(
            num_things_classes=len(dsb.things_classes),
            num_stuff_classes=len(dsb.stuff_classes),
        )
        self.datasets = self.create_dataset(
            self.cfg,
            train_annotation_path=os.path.join(
                dsb.data_dir, dsb.dataset_folder, dsb.train_ann_file
            ),
            val_annotation_path=os.path.join(
                dsb.data_dir, dsb.dataset_folder, dsb.val_ann_file
            ),
            img_dir=os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.img_folder),
            seg_dir=os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.seg_folder),
            classes=dsb.classes,
        )
        self.trainer = self.create_trainer(self.cfg, self.args.net)
        self.cfg = self.trainer.config
        if os.environ.get("WANDB_API_KEY"):
            self._register_image_hook(self.cfg, dsb)
        print("batch size:", self.cfg.data.samples_per_gpu)
        self.trainer.fit(
            model=self.model,
            train_dataset=self.datasets[0],
            val_dataset=self.datasets[1],
        )

    @staticmethod
    def _register_image_hook(cfg, dsb):
        """Build a WandbPanopticImageHook and attach it to cfg.custom_hooks."""
        # Gather image/seg paths for a few val images
        import json

        ann_path = os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.val_ann_file)
        with open(ann_path) as f:
            ann = json.load(f)
        img_dir = os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.img_folder)
        seg_dir = os.path.join(dsb.data_dir, dsb.dataset_folder, dsb.seg_folder)

        # Build coco_category_id → contiguous class index mapping
        name_to_catid = {cat["name"]: cat["id"] for cat in ann["categories"]}
        catid_to_idx = {}
        for idx, name in enumerate(dsb.classes):
            cid = name_to_catid.get(name)
            if cid is not None:
                catid_to_idx[cid] = idx

        img_paths, seg_paths, sid2idx_maps = [], [], []
        for entry in ann.get("annotations", [])[:3]:
            fname = entry["file_name"]
            img_fname = fname.replace(".png", ".jpg")
            img_paths.append(os.path.join(img_dir, img_fname))
            seg_paths.append(os.path.join(seg_dir, fname))
            # segment_id → contiguous class index (same space as model output)
            sid2idx_maps.append(
                {seg["id"]: catid_to_idx[seg["category_id"]]
                 for seg in entry["segments_info"]
                 if seg["category_id"] in catid_to_idx}
            )

        hook = WandbPanopticImageHook(
            img_paths=img_paths, seg_paths=seg_paths,
            seg_id_to_class_idx_maps=sid2idx_maps, num_images=3,
        )
        if not hasattr(cfg, "custom_hooks") or cfg.custom_hooks is None:
            cfg.custom_hooks = []
        cfg.custom_hooks.append(hook)

    def create_model(self, num_things_classes: int, num_stuff_classes: int):
        model_registry = MMPanopticSegmentationModels(
            num_things_classes=num_things_classes, num_stuff_classes=num_stuff_classes
        )
        model_dict = model_registry[self.args.net]
        model, cfg = model_dict["model"], model_dict["config"]

        return model, cfg

    def create_dataset(
        self,
        config,
        train_annotation_path,
        val_annotation_path,
        img_dir,
        seg_dir,
        classes,
    ):
        MMDatasetAdaptor.set_dataset_info_in_config(
            config,
            train_anno_file=train_annotation_path,
            val_anno_file=val_annotation_path,
            image_dir=img_dir,
            seg_dir=seg_dir,
            classes=classes,
        )

        if hasattr(config.data.train, "dataset"):
            datasets = [
                build_dataset(config.data.train.dataset),
                build_dataset(config.data.val),
            ]
        else:
            datasets = [
                build_dataset(config.data.train),
            ]
            datasets.append(build_dataset(config.data.val))

        # print(config.pretty_text)
        print("\n***** Training data:", type(datasets[0]))
        print(datasets[0])

        print("\n***** Validation data:")
        print(datasets[1])

        print("\n----------------------------- first training example:")
        print(datasets[0][0])
        return datasets

    def create_trainer(self, cfg, net: str):
        return MMTrainer(cfg, net)


if __name__ == "__main__":
    """
    python -m examples.panoptic_segmentation.mmdet.train
    """

    from argparse import ArgumentParser

    options = MMPanopticSegmentationModels.model_names()
    parser = ArgumentParser(
        epilog=f"""
            Options for net: {options} ({len(options)} of them).
            
            """
    )
    parser.add_argument("--net", type=str, default="maskrcnn_r50")
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    TrainingSession(args).run()
