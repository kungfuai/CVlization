from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import itertools
import io
import logging
import json
import os
from subprocess import check_output
import warnings
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.core.mask import BitmapMasks
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.datasets.pipelines.loading import (
    LoadAnnotations,
    PIPELINES,
    PolygonMasks,
    maskUtils,
)
from mmdet.core.evaluation import mean_ap
from terminaltables import AsciiTable
from cvlization.utils.io import download_file


LOGGER = logging.getLogger(__name__)
# Patch mmdet:
poly2mask_fn = LoadAnnotations._poly2mask
tpfp_default_fn = mean_ap.tpfp_default
eval_map_fn = mean_ap.eval_map


def new_poly2mask(self, mask, h, w):
    if isinstance(mask, np.ndarray):
        # raise ValueError(f"mask shape: {mask.shape}")
        return mask
    return poly2mask_fn(mask, h, w)


def new_eval_map(
    det_results,
    annotations,
    scale_ranges=None,
    iou_thr=0.5,
    ioa_thr=None,
    dataset=None,
    logger=None,
    tpfp_fn=None,
    nproc=4,
    use_legacy_coordinate=False,
    use_group_of=False,
):
    LOGGER.warning(f"det results: {det_results}")
    raise ValueError(f"det results: {det_results}")
    return eval_map_fn(
        det_results,
        annotations,
        scale_ranges=scale_ranges,
        iou_thr=iou_thr,
        ioa_thr=ioa_thr,
        dataset=dataset,
        logger=logger,
        tpfp_fn=tpfp_fn,
        nproc=nproc,
        use_legacy_coordinate=use_legacy_coordinate,
        use_group_of=use_group_of,
    )


def new_tpfp_default(
    det_bboxes,
    gt_bboxes,
    gt_bboxes_ignore=None,
    iou_thr=0.5,
    area_ranges=None,
    use_legacy_coordinate=False,
    **kwargs,
):
    if isinstance(det_bboxes, list):
        # This is assuming det_bboxes come from a single example
        det_bboxes = np.concatenate(det_bboxes, axis=0)
        assert det_bboxes.ndim == 2
        assert det_bboxes.shape[1] == 5
    elif isinstance(det_bboxes, np.ndarray):
        assert det_bboxes.ndim == 2, f"det_bboxes.ndim: {det_bboxes.ndim}"
        # det_bboxes = np.array(det_bboxes)
    # if not hasattr(det_bboxes, "shape"):
    #     raise ValueError(
    #         f"det_bboxes type: {type(det_bboxes)}. value = {det_bboxes}, gt_bboxes = {gt_bboxes}"
    #     )
    gt_bboxes = np.array(gt_bboxes)
    assert gt_bboxes.ndim == 2, f"gt_bboxes.ndim: {gt_bboxes.ndim}"
    return tpfp_default_fn(
        det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=gt_bboxes_ignore,
        iou_thr=iou_thr,
        area_ranges=area_ranges,
        use_legacy_coordinate=use_legacy_coordinate,
        **kwargs,
    )


# LoadAnnotations._poly2mask = new_poly2mask
# mean_ap.eval_map = new_eval_map
# mean_ap.tpfp_default = new_tpfp_default

PIPELINES.module_dict.pop("LoadAnnotations", None)


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        with_bbox=True,
        with_label=True,
        with_mask=False,
        with_seg=False,
        poly2mask=True,
        denorm_bbox=False,
        file_client_args=dict(backend="disk"),
    ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results["ann_info"]
        results["gt_bboxes"] = ann_info["bboxes"].copy()

        if self.denorm_bbox:
            bbox_num = results["gt_bboxes"].shape[0]
            if bbox_num != 0:
                h, w = results["img_shape"][:2]
                results["gt_bboxes"][:, 0::2] *= w
                results["gt_bboxes"][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get("bboxes_ignore", None)
        if gt_bboxes_ignore is not None:
            results["gt_bboxes_ignore"] = gt_bboxes_ignore.copy()
            results["bbox_fields"].append("gt_bboxes_ignore")
        results["bbox_fields"].append("gt_bboxes")

        gt_is_group_ofs = ann_info.get("gt_is_group_ofs", None)
        if gt_is_group_ofs is not None:
            results["gt_is_group_ofs"] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results["gt_labels"] = results["ann_info"]["labels"].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results["img_info"]["height"], results["img_info"]["width"]
        gt_masks = results["ann_info"]["masks"]
        if isinstance(gt_masks, np.ndarray):
            assert gt_masks.ndim == 3
            h = gt_masks.shape[1]
            w = gt_masks.shape[2]
            gt_masks = BitmapMasks(gt_masks, h, w)
        elif self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w
            )
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h, w
            )
        results["gt_masks"] = gt_masks
        results["mask_fields"].append("gt_masks")
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        img_bytes = self.file_client.get(filename)
        results["gt_semantic_seg"] = mmcv.imfrombytes(
            img_bytes, flag="unchanged"
        ).squeeze()
        results["seg_fields"].append("gt_semantic_seg")
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_mask={self.with_mask}, "
        repr_str += f"with_seg={self.with_seg}, "
        repr_str += f"poly2mask={self.poly2mask}, "
        repr_str += f"poly2mask={self.file_client_args})"
        return repr_str


@dataclass
class MMInstanceSegmentationModels:
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
    def adapt_and_register_detection_dataset(cls, dataset_builder):
        class TempCOCODataset(CocoDataset):
            def load_annotations(self, ann_file: str):
                if "train" in ann_file:
                    dataset = dataset_builder.training_dataset()
                else:
                    dataset = dataset_builder.validation_dataset()

                self.CLASSES = dataset.CLASSES
                coco_anns: dict = dataset.create_coco_annotations()
                coco_anno_file = os.path.join(
                    "/tmp", "coco", f"coco_anns_{ann_file}.json"
                )
                os.makedirs(os.path.dirname(coco_anno_file), exist_ok=True)
                with open(coco_anno_file, "w") as f:
                    json.dump(coco_anns, f)
                return super().load_annotations(coco_anno_file)

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
                for idx, row in enumerate(dataset.annotations):
                    image_path = row["image_path"]
                    image = mmcv.imread(image_path)
                    height, width = image.shape[:2]
                    relative_path = image_path.replace(
                        dataset_builder.image_dir, ""
                    ).lstrip("/")
                    data_info = dict(filename=relative_path, width=width, height=height)

                    example = dataset[idx]
                    targets = example[1]
                    if isinstance(targets, dict):
                        bboxes = targets["boxes"]
                        labels = targets["labels"]
                        masks = targets["masks"]
                    elif isinstance(targets, list):
                        bboxes, labels, masks = targets[0], targets[1], targets[2]
                    if labels.ndim == 2:
                        labels = np.squeeze(labels, -1)
                    data_anno = dict(
                        bboxes=np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                        labels=np.array(labels, dtype=np.long),
                        masks=np.array(masks, dtype=np.long),
                    )

                    data_info.update(ann=data_anno)
                    data_infos.append(data_info)
                return data_infos

            def evaluate_det_segm(
                self,
                results,
                result_files,
                coco_gt,
                metrics,
                logger=None,
                classwise=False,
                proposal_nums=(100, 300, 1000),
                iou_thrs=None,
                metric_items=None,
            ):
                """Instance segmentation and object detection evaluation in COCO
                protocol.

                Args:
                    results (list[list | tuple | dict]): Testing results of the
                        dataset.
                    result_files (dict[str, str]): a dict contains json file path.
                    coco_gt (COCO): COCO API object with ground truth annotation.
                    metric (str | list[str]): Metrics to be evaluated. Options are
                        'bbox', 'segm', 'proposal', 'proposal_fast'.
                    logger (logging.Logger | str | None): Logger used for printing
                        related information during evaluation. Default: None.
                    classwise (bool): Whether to evaluating the AP for each class.
                    proposal_nums (Sequence[int]): Proposal number used for evaluating
                        recalls, such as recall@100, recall@1000.
                        Default: (100, 300, 1000).
                    iou_thrs (Sequence[float], optional): IoU threshold used for
                        evaluating recalls/mAPs. If set to a list, the average of all
                        IoUs will also be computed. If not specified, [0.50, 0.55,
                        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                        Default: None.
                    metric_items (list[str] | str, optional): Metric items that will
                        be returned. If not specified, ``['AR@100', 'AR@300',
                        'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                        used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                        'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                        ``metric=='bbox' or metric=='segm'``.

                Returns:
                    dict[str, float]: COCO style evaluation metric.
                """
                if iou_thrs is None:
                    iou_thrs = np.linspace(
                        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
                    )
                if metric_items is not None:
                    if not isinstance(metric_items, list):
                        metric_items = [metric_items]

                eval_results = OrderedDict()
                for metric in metrics:
                    msg = f"Evaluating {metric}..."
                    if logger is None:
                        msg = "\n" + msg
                    print_log(msg, logger=logger)

                    if metric == "proposal_fast":
                        if isinstance(results[0], tuple):
                            raise KeyError(
                                "proposal_fast is not supported for "
                                "instance segmentation result."
                            )
                        ar = self.fast_eval_recall(
                            results, proposal_nums, iou_thrs, logger="silent"
                        )
                        log_msg = []
                        for i, num in enumerate(proposal_nums):
                            eval_results[f"AR@{num}"] = ar[i]
                            log_msg.append(f"\nAR@{num}\t{ar[i]:.4f}")
                        log_msg = "".join(log_msg)
                        print_log(log_msg, logger=logger)
                        continue

                    iou_type = "bbox" if metric == "proposal" else metric
                    if metric not in result_files:
                        raise KeyError(f"{metric} is not in results")
                    try:
                        predictions = mmcv.load(result_files[metric])
                        if iou_type == "segm":
                            # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                            # When evaluating mask AP, if the results contain bbox,
                            # cocoapi will use the box area instead of the mask area
                            # for calculating the instance area. Though the overall AP
                            # is not affected, this leads to different
                            # small/medium/large mask AP results.
                            for x in predictions:
                                x.pop("bbox")
                            warnings.simplefilter("once")
                            warnings.warn(
                                'The key "bbox" is deleted for more accurate mask AP '
                                "of small/medium/large instances since v2.12.0. This "
                                "does not change the overall mAP calculation.",
                                UserWarning,
                            )
                        coco_det = coco_gt.loadRes(predictions)
                    except IndexError:
                        print_log(
                            "The testing results of the whole dataset is empty.",
                            logger=logger,
                            level=logging.ERROR,
                        )
                        break

                    cocoEval = COCOeval(coco_gt, coco_det, iou_type)
                    cocoEval.params.catIds = self.cat_ids
                    cocoEval.params.imgIds = self.img_ids
                    cocoEval.params.maxDets = list(proposal_nums)
                    cocoEval.params.iouThrs = iou_thrs
                    # mapping of cocoEval.stats
                    coco_metric_names = {
                        "mAP": 0,
                        "mAP_50": 1,
                        "mAP_75": 2,
                        "mAP_s": 3,
                        "mAP_m": 4,
                        "mAP_l": 5,
                        "AR@100": 6,
                        "AR@300": 7,
                        "AR@1000": 8,
                        "AR_s@1000": 9,
                        "AR_m@1000": 10,
                        "AR_l@1000": 11,
                    }
                    if metric_items is not None:
                        for metric_item in metric_items:
                            if metric_item not in coco_metric_names:
                                raise KeyError(
                                    f"metric item {metric_item} is not supported"
                                )

                    if metric == "proposal":
                        cocoEval.params.useCats = 0
                        cocoEval.evaluate()
                        cocoEval.accumulate()

                        # Save coco summarize print information to logger
                        redirect_string = io.StringIO()
                        with contextlib.redirect_stdout(redirect_string):
                            cocoEval.summarize()
                        print_log("\n" + redirect_string.getvalue(), logger=logger)

                        if metric_items is None:
                            metric_items = [
                                "AR@100",
                                "AR@300",
                                "AR@1000",
                                "AR_s@1000",
                                "AR_m@1000",
                                "AR_l@1000",
                            ]

                        for item in metric_items:
                            val = float(
                                f"{cocoEval.stats[coco_metric_names[item]]:.3f}"
                            )
                            eval_results[item] = val
                    else:
                        cocoEval.evaluate()
                        cocoEval.accumulate()

                        # Save coco summarize print information to logger
                        redirect_string = io.StringIO()
                        with contextlib.redirect_stdout(redirect_string):
                            cocoEval.summarize()
                        print_log("\n" + redirect_string.getvalue(), logger=logger)

                        if classwise:  # Compute per-category AP
                            # Compute per-category AP
                            # from https://github.com/facebookresearch/detectron2/
                            precisions = cocoEval.eval["precision"]
                            # precision: (iou, recall, cls, area range, max dets)
                            assert len(self.cat_ids) == precisions.shape[2]

                            results_per_category = []
                            for idx, catId in enumerate(self.cat_ids):
                                # area range index 0: all area ranges
                                # max dets index -1: typically 100 per image
                                nm = self.coco.loadCats(catId)[0]
                                precision = precisions[:, :, idx, 0, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float("nan")
                                results_per_category.append(
                                    (f'{nm["name"]}', f"{float(ap):0.3f}")
                                )

                            num_columns = min(6, len(results_per_category) * 2)
                            results_flatten = list(
                                itertools.chain(*results_per_category)
                            )
                            headers = ["category", "AP"] * (num_columns // 2)
                            results_2d = itertools.zip_longest(
                                *[
                                    results_flatten[i::num_columns]
                                    for i in range(num_columns)
                                ]
                            )
                            table_data = [headers]
                            table_data += [result for result in results_2d]
                            table = AsciiTable(table_data)
                            print_log("\n" + table.table, logger=logger)

                        if metric_items is None:
                            metric_items = [
                                "mAP",
                                "mAP_50",
                                "mAP_75",
                                "mAP_s",
                                "mAP_m",
                                "mAP_l",
                            ]

                        for metric_item in metric_items:
                            key = f"{metric}_{metric_item}"
                            val = float(
                                f"{cocoEval.stats[coco_metric_names[metric_item]]:.3f}"
                            )
                            eval_results[key] = val
                        ap = cocoEval.stats[:6]
                        eval_results[f"{metric}_mAP_copypaste"] = (
                            f"{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} "
                            f"{ap[4]:.3f} {ap[5]:.3f}"
                        )

                return eval_results

            def evaluate(
                self,
                results,
                metric="bbox",
                logger=None,
                jsonfile_prefix=None,
                classwise=False,
                proposal_nums=(100, 300, 1000),
                iou_thrs=None,
                metric_items=None,
            ):
                """Evaluation in COCO protocol.

                Args:
                    results (list[list | tuple]): Testing results of the dataset.
                    metric (str | list[str]): Metrics to be evaluated. Options are
                        'bbox', 'segm', 'proposal', 'proposal_fast'.
                    logger (logging.Logger | str | None): Logger used for printing
                        related information during evaluation. Default: None.
                    jsonfile_prefix (str | None): The prefix of json files. It includes
                        the file path and the prefix of filename, e.g., "a/b/prefix".
                        If not specified, a temp file will be created. Default: None.
                    classwise (bool): Whether to evaluating the AP for each class.
                    proposal_nums (Sequence[int]): Proposal number used for evaluating
                        recalls, such as recall@100, recall@1000.
                        Default: (100, 300, 1000).
                    iou_thrs (Sequence[float], optional): IoU threshold used for
                        evaluating recalls/mAPs. If set to a list, the average of all
                        IoUs will also be computed. If not specified, [0.50, 0.55,
                        0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                        Default: None.
                    metric_items (list[str] | str, optional): Metric items that will
                        be returned. If not specified, ``['AR@100', 'AR@300',
                        'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                        used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                        'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                        ``metric=='bbox' or metric=='segm'``.

                Returns:
                    dict[str, float]: COCO style evaluation metric.
                """

                metrics = metric if isinstance(metric, list) else [metric]
                allowed_metrics = ["bbox", "segm", "proposal", "proposal_fast"]
                for metric in metrics:
                    if metric not in allowed_metrics:
                        raise KeyError(f"metric {metric} is not supported")

                coco_gt = self.coco
                self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

                result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
                eval_results = self.evaluate_det_segm(
                    results,
                    result_files,
                    coco_gt,
                    metrics,
                    logger,
                    classwise,
                    proposal_nums,
                    iou_thrs,
                    metric_items,
                )

                if tmp_dir is not None:
                    tmp_dir.cleanup()
                return eval_results

        # TempMMDataset.__name__ = "MM" + dataset_builder.__class__.__name__
        # if TempMMDataset.__name__ in DATASETS.module_dict:
        #     DATASETS.module_dict.pop(TempMMDataset.__name__)
        # DATASETS.register_module(TempMMDataset)
        TempCOCODataset.__name__ = "MM" + dataset_builder.__class__.__name__
        if TempCOCODataset.__name__ in DATASETS.module_dict:
            DATASETS.module_dict.pop(TempCOCODataset.__name__)
        DATASETS.register_module(TempCOCODataset)
        return TempCOCODataset.__name__

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

        # cfg.dataset_type = dataset_classname
        # cfg.data_root = image_dir

        # cfg.data.train.type = dataset_classname
        # cfg.data.train.data_root = image_dir
        # cfg.data.train.ann_file = "train"
        # cfg.data.train.img_prefix = ""

        # cfg.data.val.type = dataset_classname
        # cfg.data.val.data_root = image_dir
        # cfg.data.val.ann_file = "val"
        # cfg.data.val.img_prefix = ""

        # if hasattr(cfg.data.train, "dataset"):
        #     cfg.data.train.dataset.type = dataset_classname
        #     cfg.data.train.dataset.data_root = image_dir
        #     cfg.data.train.dataset.ann_file = "train"
        #     cfg.data.train.dataset.img_prefix = ""

        # if hasattr(cfg.data.val, "dataset"):
        #     cfg.data.val = cfg.data.val.dataset
        #     cfg.data.val.pop("dataset", None)
        #     # cfg.data.val.dataset.type = dataset_classname
        #     # cfg.data.val.dataset.data_root = image_dir
        #     # cfg.data.val.dataset.ann_file = "val"
        #     # cfg.data.val.dataset.img_prefix = ""

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
