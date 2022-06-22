from dataclasses import dataclass
import logging
import json
import os
from subprocess import check_output
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)


@dataclass
class MMInstanceSegmentationModels:
    version: str = "2.24.1"  # this is the version of mmdetection
    data_dir: str = "./data"
    num_classes: int = 3
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

        if hasattr(config.model, "roi_head"):
            bbox_head = config.model.roi_head.bbox_head
            if isinstance(bbox_head, list):
                for h in bbox_head:
                    h["num_classes"] = self.num_classes
            else:
                bbox_head.num_classes = self.num_classes

            mask_head = config.model.roi_head.mask_head
            if isinstance(mask_head, list):
                for h in mask_head:
                    h["num_classes"] = self.num_classes
            else:
                mask_head.num_classes = self.num_classes
        else:
            config.model.bbox_head.num_classes = self.num_classes

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
        "config_path": "configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py",
        # "configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
        # "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
    },
    "queryinst": {  # for instance segmentation
        "config_path": "configs/queryinst/queryinst_r50_fpn_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth",
    },
}


class MMDatasetAdaptor:
    @classmethod
    def set_dataset_info_in_config(cls, cfg, dataset_builder, image_dir):
        cfg.dataset_type = "COCODataset"

        train_ds = dataset_builder.training_dataset()
        val_ds = dataset_builder.validation_dataset()
        coco_anns: dict = train_ds.create_coco_annotations()
        coco_anno_file_train = os.path.join("/tmp", "coco", f"coco_anns_train.json")
        os.makedirs(os.path.dirname(coco_anno_file_train), exist_ok=True)
        with open(coco_anno_file_train, "w") as f:
            json.dump(coco_anns, f)
        coco_anns: dict = val_ds.create_coco_annotations()
        coco_anno_file_val = os.path.join("/tmp", "coco", f"coco_anns_val.json")
        os.makedirs(os.path.dirname(coco_anno_file_val), exist_ok=True)
        with open(coco_anno_file_val, "w") as f:
            json.dump(coco_anns, f)

        cfg.data.test.ann_file = coco_anno_file_val
        cfg.data.test.img_prefix = image_dir
        cfg.data.test.classes = tuple(train_ds.CLASSES)

        cfg.data.train.ann_file = coco_anno_file_train
        cfg.data.train.img_prefix = image_dir
        cfg.data.train.classes = tuple(train_ds.CLASSES)

        cfg.data.val.ann_file = coco_anno_file_val
        cfg.data.val.img_prefix = image_dir
        cfg.data.val.classes = tuple(train_ds.CLASSES)

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
