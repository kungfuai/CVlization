"""Panoptic segmentation models provided by MMDetection.
"""

from dataclasses import dataclass
import logging
import os
from subprocess import check_output
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)


@dataclass
class MMPanopticSegmentationModels:
    version: str = "2.24.1"  # this is the version of mmdetection
    data_dir: str = "./data"
    num_things_classes: int = 3
    num_stuff_classes: int = 3
    device: str = "cuda"
    work_dir = "./tmp"

    @classmethod
    def model_names(cls) -> list:
        return list(MODEL_MENU.keys())

    def __getitem__(self, model_menu_key: str):
        if not self.loaded:
            self.load()
        config_path = MODEL_MENU[model_menu_key]["config_path"]
        checkpoint_url = MODEL_MENU[model_menu_key]["checkpoint_url"]
        config_path = os.path.join(
            self.data_dir, f"mmdetection-{self.version}", config_path
        )

        config = mmcv.Config.fromfile(config_path)
        config.work_dir = self.work_dir
        config.device = self.device

        checkpoint_filepath = os.path.join(
            self.data_dir, "checkpoints", "mmdetection", checkpoint_url.split("/")[-1]
        )
        config.load_from = checkpoint_filepath
        if not os.path.isfile(checkpoint_filepath):
            print("Downloading checkpoint...")
            download_file(checkpoint_url, checkpoint_filepath)

        print(config.pretty_text)
        # assert False
        if hasattr(config.model, "roi_head"):
            bbox_head = config.model.roi_head.bbox_head
            if isinstance(bbox_head, list):
                for h in bbox_head:
                    h["num_classes"] = self.num_things_classes
            else:
                bbox_head.num_classes = self.num_things_classes

            mask_head = config.model.roi_head.mask_head
            if isinstance(mask_head, list):
                for h in mask_head:
                    h["num_classes"] = self.num_things_classes
            else:
                mask_head.num_classes = self.num_things_classes
        if hasattr(config.model, "semantic_head"):
            config.model.semantic_head.num_things_classes = self.num_things_classes
            config.model.semantic_head.num_stuff_classes = self.num_stuff_classes
        if hasattr(config.model, "panoptic_head"):
            config.model.panoptic_head.num_things_classes = self.num_things_classes
            config.model.panoptic_head.num_stuff_classes = self.num_stuff_classes
        if hasattr(config.model, "panoptic_fusion_head"):
            config.model.panoptic_fusion_head.num_things_classes = (
                self.num_things_classes
            )
            config.model.panoptic_fusion_head.num_stuff_classes = self.num_stuff_classes

        # Initialize the detector
        model = build_detector(config.model)

        # Load checkpoint
        checkpoint = load_checkpoint(
            model, checkpoint_filepath, map_location=self.device
        )

        # Set the classes of models for inference
        model.CLASSES = checkpoint["meta"]["CLASSES"]

        # We need to set the model's cfg for inference
        model.cfg = config

        # Convert the model to GPU
        model.to(self.device)
        # Convert the model into evaluation mode
        model = model.eval()

        return dict(config=config, checkpoint_path=checkpoint_filepath, model=model)

    def load(self):
        if not self._is_extracted():
            self.extract()
        self.loaded = True

    def __post_init__(self):
        self.loaded = False

    @property
    def package_download_url(self):
        return f"https://github.com/open-mmlab/mmdetection/archive/refs/tags/v{self.version}.tar.gz"

    @property
    def download_filepath(self):
        download_filename = "mmdetection-" + self.package_download_url.split("/")[-1]
        return os.path.join(self.data_dir, download_filename)

    def download(self):
        check_output("mkdir -p ./data".split())
        check_output(
            f"wget {self.package_download_url} -O {self.download_filepath}".split()
        )
        assert os.path.isfile(
            self.download_filepath
        ), f"{self.download_filepath} not found."

    def extract(self):
        if not self._is_downloaded():
            self.download()
        check_output(f"tar -xzf {self.download_filepath} -C {self.data_dir}".split())

    def _is_downloaded(self):
        return os.path.isfile(self.download_filepath)

    def _is_extracted(self):
        return os.path.isdir(self.download_filepath.replace(".tar.gz", ""))


MODEL_MENU = {
    "maskrcnn_r50": {
        "config_path": "configs/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py",
        # https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth",
    },
    "maskformer_r50": {
        "config_path": "configs/maskformer/maskformer_r50_mstrain_16x1_75e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/maskformer/maskformer_r50_mstrain_16x1_75e_coco/maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth",
    },
    "maskrcnn_swin": {
        "config_path": "configs/maskformer/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/maskformer/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco_20220326_221612-061b4eb8.pth",
    },
    "mask2former_r50": {
        "config_path": "configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth",
    },
    "mask2former_swin_t": {
        "config_path": "configs/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco-panoptic_20220326_224553-fc567107.pth",
    },
}


class MMDatasetAdaptor:
    @classmethod
    def set_dataset_info_in_config(
        cls, cfg, train_anno_file, val_anno_file, image_dir, seg_dir, classes
    ):
        cfg.dataset_type = "COCOPanopticDataset"

        cfg.data.test.ann_file = val_anno_file
        cfg.data.test.img_prefix = image_dir
        cfg.data.test.seg_prefix = seg_dir
        cfg.data.test.classes = classes

        cfg.data.train.ann_file = train_anno_file
        cfg.data.train.img_prefix = image_dir
        cfg.data.train.seg_prefix = seg_dir
        cfg.data.train.classes = classes

        cfg.data.val.ann_file = val_anno_file
        cfg.data.val.img_prefix = image_dir
        cfg.data.val.seg_prefix = seg_dir
        cfg.data.val.classes = classes

        return cfg


class MMTrainer:
    def __init__(self, config, net: str):
        self.config = config
        self.net = net
        self.configure()

    def fit(self, model, train_dataset, val_dataset=None, **kwargs):
        print("batch size:", self.config.data.samples_per_gpu)
        model.CLASSES = train_dataset.CLASSES
        train_detector(
            model,
            # `mmdet`` uses config.data.val to construct val_dataset, and pass it to EvalHook
            # So it is not necessary and also not useful to provide val_dataset here.
            [train_dataset],
            self.config,
            distributed=False,
            validate=True,
            **kwargs,
        )

    def configure(self):
        cfg = self.config
        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        cfg.optimizer.lr = 0.02 / 8
        cfg.lr_config.warmup = None
        cfg.log_config.interval = 10

        cfg.evaluation.metric == ["bbox", "segm"]

        # We can set the evaluation interval to reduce the evaluation times
        cfg.evaluation.interval = 1
        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = 10
        if hasattr(cfg.data.train, "times"):
            cfg.data.train.pop("times", None)

        # Set seed thus the results are more reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=False)
        cfg.gpu_ids = range(1)

        # We can also use tensorboard to log the training process
        cfg.log_config.hooks = [
            dict(type="TextLoggerHook"),
            # dict(type="TensorboardLoggerHook"),
        ]

        cfg.runner.max_epochs = 15

        model_menu_key = self.net
        if "retinanet" in model_menu_key:
            cfg.optimizer.lr = 0.001 / 8
            cfg.lr_config.warmup = "linear"
            cfg.lr_config.warmup_iters = 200
            cfg.lr_config.warmup_ratio = 0.001
            cfg.runner.max_epochs = 35
        elif "detr" in model_menu_key:
            cfg.optimizer.lr = 0.002 / 8
            cfg.lr_config.warmup = "linear"
            cfg.lr_config.warmup_iters = 100
            cfg.lr_config.warmup_ratio = 0.001
            cfg.runner.max_epochs = 35
        elif "dyhead" in model_menu_key:
            cfg.optimizer.lr = 0.00001
            cfg.lr_config.warmup = "linear"
            cfg.lr_config.warmup_iters = 200
            cfg.lr_config.warmup_ratio = 0.001
            cfg.runner.max_epochs = 35
