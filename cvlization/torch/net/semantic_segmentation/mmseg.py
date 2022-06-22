"""Panoptic segmentation models provided by MMDetection.
"""

from dataclasses import dataclass
import logging
import os
from subprocess import check_output
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import set_random_seed, train_segmentor, init_segmentor
from mmseg.models import build_segmentor
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
        config.norm_cfg = dict(type="BN", requires_grad=True)
        config.model.backbone.norm_cfg = config.norm_cfg
        config.model.decode_head.norm_cfg = config.norm_cfg
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


MODEL_MENU = {
    "pspnet_r50": {
        "config_path": "configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint_url": "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
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
