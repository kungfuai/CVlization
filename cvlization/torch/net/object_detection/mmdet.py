from dataclasses import dataclass
import os
from subprocess import check_output
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import build_dataset
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed, train_detector
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from cvlization.utils.io import download_file
from cvlization.lab.kitti_tiny import KittiTinyDatasetBuilder


@dataclass
class MMDetectionModels:
    version: str = "2.24.1"
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
    # yolox seems to require image size to be fixed!
    # RecursionError: maximum recursion depth exceeded
    # "yolox_tiny_416": {
    #     "config_path": "configs/yolox/yolox_tiny_8x8_300e_coco.py",
    #     "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
    # },
    # "yolox_s_640": {
    #     "config_path": "configs/yolox/yolox_s_8x8_300e_coco.py",
    #     "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth",
    # },
    "retinanet_r18": {
        "config_path": "configs/retinanet/retinanet_r18_fpn_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r18_fpn_1x_coco/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth",
    },
    "retinanet_r50_caffe": {
        "config_path": "configs/retinanet/retinanet_r50_caffe_fpn_mstrain_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth",
    },
    "retinanet_r50": {
        "config_path": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
    },
    "faster_rcnn": {
        "config_path": "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth",
    },
    "fcos": {
        "config_path": "configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
    },
    "centernet": {
        "config_path": "configs/centernet/centernet_resnet18_dcnv2_140e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth",
    },
    "cornernet": {
        # gpu out of memory (tried to allocate 48G)
        "config_path": "configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco/cornernet_hourglass104_mstest_32x3_210e_coco_20200819_203110-1efaea91.pth",
    },
    "deformable_detr": {
        "config_path": "configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth",
    },
    "detr": {
        "config_path": "configs/detr/detr_r50_8x2_150e_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth",
    },
    "dyhead": {
        "config_path": "configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py",
        "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_r50_fpn_dyhead_4x4_1x_coco/atss_r50_fpn_dyhead_4x4_1x_coco_20211219_023314-eaa620c6.pth",
    },
    # "queryinst": { # for instance segmentation
    #     "config_path": "configs/queryinst/queryinst_r50_fpn_1x_coco.py",
    #     "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"
    # },
}


class MMDatasetAdaptor:
    @classmethod
    def adapt_and_register_detection_dataset(cls, dataset_builder):
        class TempMMDataset(CustomDataset):
            def load_annotations(self, ann_file: str):
                if "train" in ann_file:
                    dataset = dataset_builder.training_dataset()
                else:
                    dataset = dataset_builder.validation_dataset()

                self._base_dataset = dataset

                self.CLASSES = dataset.CLASSES
                data_infos = []
                assert hasattr(dataset, "annotations")
                if dataset.annotations is None:
                    dataset.load_annotations()
                    assert (
                        dataset.annotations is not None
                    ), f"{dataset} has no annotations"
                for row in dataset.annotations:
                    image_path = row["image_path"]
                    image = mmcv.imread(image_path)
                    height, width = image.shape[:2]
                    relative_path = image_path.replace(
                        dataset_builder.image_dir, ""
                    ).lstrip("/")
                    data_info = dict(filename=relative_path, width=width, height=height)
                    data_anno = dict(
                        bboxes=np.array(row["bboxes"], dtype=np.float32).reshape(-1, 4),
                        labels=np.array(row["labels"], dtype=np.long),
                        bboxes_ignore=np.array(
                            row["bboxes_ignore"], dtype=np.float32
                        ).reshape(-1, 4),
                        labels_ignore=np.array(row["labels_ignore"], dtype=np.long),
                    )
                    data_info.update(ann=data_anno)
                    data_infos.append(data_info)
                return data_infos

        TempMMDataset.__name__ = "MM" + dataset_builder.__class__.__name__
        if TempMMDataset.__name__ in DATASETS.module_dict:
            DATASETS.module_dict.pop(TempMMDataset.__name__)
        DATASETS.register_module(TempMMDataset)
        return TempMMDataset.__name__

    @classmethod
    def set_dataset_info_in_config(cls, cfg, dataset_classname, image_dir):
        cfg.dataset_type = dataset_classname
        cfg.data_root = image_dir

        cfg.data.train.type = dataset_classname
        cfg.data.train.data_root = image_dir
        cfg.data.train.ann_file = "train"
        cfg.data.train.img_prefix = ""

        cfg.data.val.type = dataset_classname
        cfg.data.val.data_root = image_dir
        cfg.data.val.ann_file = "val"
        cfg.data.val.img_prefix = ""

        if hasattr(cfg.data.train, "dataset"):
            cfg.data.train.dataset.type = dataset_classname
            cfg.data.train.dataset.data_root = image_dir
            cfg.data.train.dataset.ann_file = "train"
            cfg.data.train.dataset.img_prefix = ""

        if hasattr(cfg.data.val, "dataset"):
            cfg.data.val = cfg.data.val.dataset
            cfg.data.val.pop("dataset", None)
            # cfg.data.val.dataset.type = dataset_classname
            # cfg.data.val.dataset.data_root = image_dir
            # cfg.data.val.dataset.ann_file = "val"
            # cfg.data.val.dataset.img_prefix = ""

        return cfg


class MMDetectionTrainer:
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

        # Change the evaluation metric since we use customized dataset.
        cfg.evaluation.metric = "mAP"
        # We can set the evaluation interval to reduce the evaluation times
        cfg.evaluation.interval = 5
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
