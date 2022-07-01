"""Panoptic segmentation models provided by MMSegmentation.
"""

from dataclasses import dataclass
import logging
import os
from subprocess import check_output
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import set_random_seed, train_segmentor, init_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)


@dataclass
class MMSemanticSegmentationModels:
    version: str = "0.25.0"  # this is the version of mmseg to download configs from
    data_dir: str = "./data"
    num_classes: int = 3
    device: str = "cuda"
    work_dir = "./tmp"
    package_name = "mmsegmentation"

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
        model = build_segmentor(config.model)

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


"""
For vit:

 File "/home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/mmseg/models/backbones/vit.py", line 121, in forward
    x = _inner_forward(x)
  File "/home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/mmseg/models/backbones/vit.py", line 114, in _inner_forward
    x = self.attn(self.norm1(x), identity=x)
  File "/home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/torch/nn/modules/batchnorm.py", line 135, in forward
    self._check_input_dim(input)
  File "/home/ubuntu/anaconda3/envs/tensorflow2_p38/lib/python3.8/site-packages/torch/nn/modules/batchnorm.py", line 407, in _check_input_dim
    raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
ValueError: expected 4D input (got 3D input)
"""

MODEL_MENU = {
    "pspnet_r50": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet
        "config_path": "configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
    },
    "deeplabv3_r18": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3
        "config_path": "configs/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth",
    },
    "deeplabv3_r50": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3
        "config_path": "configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth",
    },
    "apcnet_r50": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet
        "config_path": "configs/apcnet/apcnet_r50-d8_769x769_40k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_40k_cityscapes/apcnet_r50-d8_769x769_40k_cityscapes_20201214_115717-2a2628d7.pth",
    },
    "lraspp_mobilenetv3": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mobilenet_v3
        "config_path": "configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes/lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth",
    },
    "unet_d16_fcn": {
        # OOM!
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/README.md
        "config_path": "configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth",
    },
    "upernet_r50": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py
        "config_path": "configs/upernet/upernet_r50_512x1024_40k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth",
    },
    "upernet_vit_b": {
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/README.md
        "config_path": "configs/vit/upernet_vit-b16_mln_512x512_80k_ade20k.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1.pth",
    },
    "upernet_swin_s": {
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/README.md
        "config_path": "configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth",
    },
    "upernet_swin_b": {
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py
        "config_path": "configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth",
    },
    "segformer_b0": {
        # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer
        # https://github.com/open-mmlab/mmsegmentation/blob/master/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py
        "config_path": "configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py",
        "checkpoint_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
    }
    # TODO: test https://github.com/open-mmlab/mmsegmentation/tree/master/configs/twins
    # TODO: test https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter
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
        train_segmentor(
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
