"""Pose estimation (keypoint detection) models provided by MMPose.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
from os import path as osp
import numpy as np
import tempfile
from subprocess import check_output
import mmcv
from mmcv.runner import set_random_seed
from mmpose.apis import train_model
from mmpose.models import build_posenet
from mmpose.core.evaluation.top_down_eval import keypoint_nme, keypoint_pck_accuracy
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)


@dataclass
class MMPoseModels:
    version: str = "0.27.0"  # this is the version of mmseg to download configs from
    data_dir: str = "./data"
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

        # print(config.pretty_text)

        # head_fields = ["decoder_head", "auxiliary_head"]
        # for head_field in head_fields:
        #     if hasattr(config.model, head_field):
        #         head = getattr(config.model, head_field)
        #         head.num_classes = self.num_classes

        # Initialize the detector
        model = build_posenet(config.model)

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
    "pose_hrnet_w32": {
        # from https://github.com/open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb
        # "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py",
        "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    },
    "pose_hrnet_w48": {
        "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
    },
    "pose_swin_t": {
        "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_t_p4_w7_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192-eaefe010_20220503.pth",
    },
    # The following are from https://mmpose.readthedocs.io/en/latest/papers/datasets.html
    "deeppose_resnet_50": {
        "config_path": "configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth",
    },
    "pose_mobilenetv2": {
        "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth",
    },
    "mspn_50": {
        "config_path": "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mspn50_coco_256x192.py",
        "checkpoint_url": "https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192-8fbfb5d0_20201123.pth",
    },
}


class MMDatasetAdaptor:
    @classmethod
    def adapt_and_register_detection_dataset(cls, dataset_builder):
        class TempMMPoseDataset(Kpt2dSviewRgbImgTopDownDataset):
            def __init__(
                self,
                ann_file: str,
                img_prefix,
                data_cfg,
                pipeline,
                dataset_info=None,
                test_mode=False,
            ):
                ann_file_basename = os.path.basename(ann_file)
                if ann_file_basename == dataset_builder.train_ann_file:
                    self.base_dataset = dataset_builder.training_dataset()
                elif ann_file_basename == dataset_builder.val_ann_file:
                    self.base_dataset = dataset_builder.validation_dataset()
                else:
                    raise ValueError(f"Unknown ann_file: {ann_file_basename}")

                assert os.path.isfile(ann_file), f"{ann_file} not found."
                super().__init__(
                    ann_file,
                    img_prefix,
                    data_cfg,
                    pipeline,
                    dataset_info,
                    coco_style=False,
                    test_mode=test_mode,
                )
                self.ann_info = {**self.base_dataset.ann_info, **self.ann_info}
                # assert os.path.exists(self.img_dir) and self.split is not None
                # self.ann_info = self.base_dataset.ann_info
                self.dataset_name = "coco_pose_tiny"
                self.db = self.base_dataset.annotations

            def _xywh2cs(self, x, y, w, h):
                """This encodes bbox(x, y, w, h) into (center, scale)
                Args:
                    x, y, w, h
                Returns:
                    tuple: A tuple containing center and scale.
                    - center (np.ndarray[float32](2,)): center of the bbox (x, y).
                    - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
                """
                aspect_ratio = (
                    self.ann_info["image_size"][0] / self.ann_info["image_size"][1]
                )
                center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
                if w > aspect_ratio * h:
                    h = w * 1.0 / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio

                # pixel std is 200.0
                scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
                # padding to include proper amount of context
                scale = scale * 1.25
                return center, scale

            def evaluate(self, results, res_folder=None, metric="PCK", **kwargs):
                """Evaluate keypoint detection results. The pose prediction results will
                be saved in `${res_folder}/result_keypoints.json`.

                Note:
                batch_size: N
                num_keypoints: K
                heatmap height: H
                heatmap width: W

                Args:
                results (list(preds, boxes, image_path, output_heatmap))
                    :preds (np.ndarray[N,K,3]): The first two dimensions are
                        coordinates, score is the third dimension of the array.
                    :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                        , scale[1],area, score]
                    :image_paths (list[str]): For example, ['Test/source/0.jpg']
                    :output_heatmap (np.ndarray[N, K, H, W]): model outputs.

                res_folder (str, optional): The folder to save the testing
                        results. If not specified, a temp folder will be created.
                        Default: None.
                metric (str | list[str]): Metric to be performed.
                    Options: 'PCK', 'NME'.

                Returns:
                    dict: Evaluation results for evaluation metric.
                """
                metrics = metric if isinstance(metric, list) else [metric]
                allowed_metrics = ["PCK", "NME"]
                for metric in metrics:
                    if metric not in allowed_metrics:
                        raise KeyError(f"metric {metric} is not supported")

                if res_folder is not None:
                    tmp_folder = None
                    res_file = osp.join(res_folder, "result_keypoints.json")
                else:
                    tmp_folder = tempfile.TemporaryDirectory()
                    res_file = osp.join(tmp_folder.name, "result_keypoints.json")

                kpts = []
                for result in results:
                    preds = result["preds"]
                    boxes = result["boxes"]
                    image_paths = result["image_paths"]
                    bbox_ids = result["bbox_ids"]

                    batch_size = len(image_paths)
                    for i in range(batch_size):
                        kpts.append(
                            {
                                "keypoints": preds[i].tolist(),
                                "center": boxes[i][0:2].tolist(),
                                "scale": boxes[i][2:4].tolist(),
                                "area": float(boxes[i][4]),
                                "score": float(boxes[i][5]),
                                "bbox_id": bbox_ids[i],
                            }
                        )
                kpts = self._sort_and_unique_bboxes(kpts)

                self._write_keypoint_results(kpts, res_file)
                info_str = self._report_metric(res_file, metrics)
                name_value = OrderedDict(info_str)

                if tmp_folder is not None:
                    tmp_folder.cleanup()

                return name_value

            def _get_db(self, *args, **kwargs):
                return self.db

            def _report_metric(self, res_file, metrics, pck_thr=0.3):
                """Keypoint evaluation.

                Args:
                res_file (str): Json file stored prediction results.
                metrics (str | list[str]): Metric to be performed.
                    Options: 'PCK', 'NME'.
                pck_thr (float): PCK threshold, default: 0.3.

                Returns:
                dict: Evaluation results for evaluation metric.
                """
                info_str = []

                with open(res_file, "r") as fin:
                    preds = json.load(fin)
                assert len(preds) == len(self.db)

                outputs = []
                gts = []
                masks = []

                for pred, item in zip(preds, self.db):
                    outputs.append(np.array(pred["keypoints"])[:, :-1])
                    gts.append(np.array(item["joints_3d"])[:, :-1])
                    masks.append((np.array(item["joints_3d_visible"])[:, 0]) > 0)

                outputs = np.array(outputs)
                gts = np.array(gts)
                masks = np.array(masks)

                normalize_factor = self._get_normalize_factor(gts)

                if "PCK" in metrics:
                    _, pck, _ = keypoint_pck_accuracy(
                        outputs, gts, masks, pck_thr, normalize_factor
                    )
                    info_str.append(("PCK", pck))

                if "NME" in metrics:
                    info_str.append(
                        ("NME", keypoint_nme(outputs, gts, masks, normalize_factor))
                    )

                return info_str

            @staticmethod
            def _write_keypoint_results(keypoints, res_file):
                """Write results into a json file."""

                with open(res_file, "w") as f:
                    json.dump(keypoints, f, sort_keys=True, indent=4)

            @staticmethod
            def _sort_and_unique_bboxes(kpts, key="bbox_id"):
                """sort kpts and remove the repeated ones."""
                kpts = sorted(kpts, key=lambda x: x[key])
                num = len(kpts)
                for i in range(num - 1, 0, -1):
                    if kpts[i][key] == kpts[i - 1][key]:
                        del kpts[i]

                return kpts

            @staticmethod
            def _get_normalize_factor(gts):
                """Get inter-ocular distance as the normalize factor, measured as the
                Euclidean distance between the outer corners of the eyes.

                Args:
                    gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

                Return:
                    np.ndarray[N, 2]: normalized factor
                """

                interocular = np.linalg.norm(
                    gts[:, 0, :] - gts[:, 1, :], axis=1, keepdims=True
                )
                return np.tile(interocular, [1, 2])

        TempMMPoseDataset.__name__ = "MM" + dataset_builder.__class__.__name__
        if TempMMPoseDataset.__name__ in DATASETS.module_dict:
            DATASETS.module_dict.pop(TempMMPoseDataset.__name__)
        DATASETS.register_module(TempMMPoseDataset)
        return TempMMPoseDataset.__name__

    @classmethod
    def set_dataset_info_in_config(
        cls,
        cfg,
        dataset_classname,
        dataset_dir,
        train_anno_file,
        val_anno_file,
        image_dir,
    ):
        cfg.data_root = dataset_dir

        # cfg.img_norm_cfg = dict(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
        # )
        cfg.data.val_dataloader = dict(samples_per_gpu=2)
        cfg.data.test_dataloader = dict(samples_per_gpu=2)

        cfg.data.train.type = dataset_classname
        cfg.data.train.img_prefix = os.path.join(dataset_dir, image_dir)
        cfg.data.train.ann_file = os.path.join(dataset_dir, train_anno_file)

        cfg.data.val.type = dataset_classname
        cfg.data.val.img_prefix = os.path.join(dataset_dir, image_dir)
        cfg.data.val.ann_file = os.path.join(dataset_dir, val_anno_file)

        cfg.data.test.type = dataset_classname
        cfg.data.test.img_prefix = os.path.join(dataset_dir, image_dir)
        cfg.data.test.ann_file = os.path.join(dataset_dir, val_anno_file)

        return cfg


class MMTrainer:
    def __init__(self, config, net: str):
        self.config = config
        self.net = net
        self.configure()

    def fit(self, model, train_dataset, val_dataset=None, **kwargs):
        print("batch size:", self.config.data.samples_per_gpu)
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

        cfg.evaluation.metric = "PCK"
        cfg.evaluation.save_best = "PCK"

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

        cfg.total_epochs = 50
