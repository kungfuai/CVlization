"""Pose estimation (keypoint detection) models provided by MMPose.
"""

from dataclasses import dataclass
import logging
import os
from subprocess import check_output
import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    vis_pose_result,
    process_mmdet_results,
    train_model,
)
from mmpose.models import build_posenet
from mmpose.core.evaluation.top_down_eval import keypoint_nme, keypoint_pck_accuracy
from mmpose.datasets.builder import DATASETS
from mmdet.apis import inference_detector, init_detector
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)


@dataclass
class MMPoseModels:
    version: str = "0.27.0"  # this is the version of mmseg to download configs from
    data_dir: str = "./data"
    num_classes: int = 3
    device: str = "cuda"
    work_dir = "./tmp"
    package_name = "mmpose"

    @classmethod
    def model_names(cls) -> list:
        return list(MODEL_MENU.keys())

    def __getitem__(self, model_menu_key: str):
        if not self.loaded:
            self.load()
        config_path = MODEL_MENU[model_menu_key]["config_path"]
        checkpoint_url = MODEL_MENU[model_menu_key]["checkpoint_url"]
        config_path = os.path.join(
            self.data_dir, f"{self.package_name}-{self.version}", config_path
        )

        config = mmcv.Config.fromfile(config_path)
        config.work_dir = self.work_dir
        config.device = self.device

        checkpoint_filepath = os.path.join(
            self.data_dir,
            "checkpoints",
            self.package_name,
            checkpoint_url.split("/")[-1],
        )
        config.load_from = checkpoint_filepath
        if not os.path.isfile(checkpoint_filepath):
            print("Downloading checkpoint...")
            download_file(checkpoint_url, checkpoint_filepath)

        print(config.pretty_text)
        head_fields = ["decoder_head", "auxiliary_head"]
        for head_field in head_fields:
            if hasattr(config.model, head_field):
                head = getattr(config.model, head_field)
                head.num_classes = self.num_classes

        # Since we use only one GPU, BN is used instead of SyncBN
        # TODO: revisit norm type!
        config.norm_cfg = dict(type="BN", requires_grad=True)
        # config.norm_cfg = dict(type="LN", requires_grad=True)
        config.model.backbone.norm_cfg = config.norm_cfg
        config.model.decode_head.norm_cfg = config.norm_cfg
        if hasattr(config.model, "auxiliary_head"):
            config.model.auxiliary_head.norm_cfg = config.norm_cfg

        # Initialize the detector
        model = build_posenet(config.model)

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
        return f"https://github.com/open-mmlab/{self.package_name}/archive/refs/tags/v{self.version}.tar.gz"

    @property
    def download_filepath(self):
        download_filename = (
            f"{self.package_name}-" + self.package_download_url.split("/")[-1]
        )
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
    "pspnet_r50": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet
        "config_path": "configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
    },
}


class MMDatasetAdaptor:
    @classmethod
    def adapt_and_register_detection_dataset(cls, dataset_builder):
        class TempMMSegDataset(CustomDataset):
            CLASSES = dataset_builder.CLASSES
            PALETTE = dataset_builder.PALETTE

            def __init__(self, split, **kwargs):
                super().__init__(
                    img_suffix=".jpg", seg_map_suffix=".png", split=split, **kwargs
                )
                assert os.path.exists(self.img_dir) and self.split is not None

        TempMMSegDataset.__name__ = "MM" + dataset_builder.__class__.__name__
        if TempMMSegDataset.__name__ in DATASETS.module_dict:
            DATASETS.module_dict.pop(TempMMSegDataset.__name__)
        DATASETS.register_module(TempMMSegDataset)
        return TempMMSegDataset.__name__

    @classmethod
    def set_dataset_info_in_config(
        cls,
        cfg,
        dataset_classname,
        dataset_dir,
        train_anno_file,
        val_anno_file,
        image_dir,
        seg_dir,
    ):
        cfg.dataset_type = dataset_classname

        cfg.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
        )
        # cfg.crop_size = (256, 256)

        # TODO: pass the data pipeline as parameters.
        cfg.train_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=(320, 240), ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.75),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **cfg.img_norm_cfg),
            dict(type="Pad", size=cfg.crop_size, pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ]

        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(320, 240),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", **cfg.img_norm_cfg),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ]

        cfg.data.train.type = cfg.dataset_type
        cfg.data.train.data_root = dataset_dir
        cfg.data.train.img_dir = image_dir
        cfg.data.train.ann_dir = seg_dir
        cfg.data.train.pipeline = cfg.train_pipeline
        cfg.data.train.split = train_anno_file

        cfg.data.val.type = cfg.dataset_type
        cfg.data.val.data_root = dataset_dir
        cfg.data.val.img_dir = image_dir
        cfg.data.val.ann_dir = seg_dir
        cfg.data.val.pipeline = cfg.test_pipeline
        cfg.data.val.split = val_anno_file

        cfg.data.test.type = cfg.dataset_type
        cfg.data.test.data_root = dataset_dir
        cfg.data.test.img_dir = image_dir
        cfg.data.test.ann_dir = seg_dir
        cfg.data.test.pipeline = cfg.test_pipeline
        cfg.data.test.split = val_anno_file

        return cfg


class MMTrainer:
    def __init__(self, config, net: str):
        self.config = config
        self.net = net
        self.configure()

    def fit(self, model, train_dataset, val_dataset=None, **kwargs):
        print("batch size:", self.config.data.samples_per_gpu)
        model.CLASSES = train_dataset.CLASSES
        train_model(
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
        # cfg.optimizer.lr = 0.02 / 8
        # cfg.lr_config.warmup = None
        cfg.log_config.interval = 10

        # We can set the evaluation interval to reduce the evaluation times
        cfg.evaluation.interval = 2
        # We can set the checkpoint saving interval to reduce the storage cost
        cfg.checkpoint_config.interval = 50
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

        cfg.runner.max_iters = 50
