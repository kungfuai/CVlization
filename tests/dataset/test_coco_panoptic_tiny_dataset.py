import importlib
import sys
import types

import numpy as np
from PIL import Image


def _install_stub_dependencies():
    """Install lightweight stubs so the dataset module can be imported in unit tests."""
    panopticapi_mod = types.ModuleType("panopticapi")
    panoptic_utils_mod = types.ModuleType("panopticapi.utils")

    def rgb2id(pan_png):
        pan_png = np.asarray(pan_png)
        if pan_png.ndim == 2:
            return pan_png.astype(np.int64)
        return (
            pan_png[..., 0].astype(np.int64)
            + 256 * pan_png[..., 1].astype(np.int64)
            + 256 * 256 * pan_png[..., 2].astype(np.int64)
        )

    panoptic_utils_mod.rgb2id = rgb2id
    panopticapi_mod.utils = panoptic_utils_mod

    mmdet_mod = types.ModuleType("mmdet")
    mmdet_datasets_mod = types.ModuleType("mmdet.datasets")
    mmdet_coco_panoptic_mod = types.ModuleType("mmdet.datasets.coco_panoptic")

    class PlaceholderCOCOPanoptic:
        def __init__(self, *args, **kwargs):
            self.dataset = {"categories": []}
            self.imgs = {}
            self.cats = {}
            self.anns = {}

    mmdet_coco_panoptic_mod.COCOPanoptic = PlaceholderCOCOPanoptic
    mmdet_datasets_mod.coco_panoptic = mmdet_coco_panoptic_mod
    mmdet_mod.datasets = mmdet_datasets_mod

    sys.modules["panopticapi"] = panopticapi_mod
    sys.modules["panopticapi.utils"] = panoptic_utils_mod
    sys.modules["mmdet"] = mmdet_mod
    sys.modules["mmdet.datasets"] = mmdet_datasets_mod
    sys.modules["mmdet.datasets.coco_panoptic"] = mmdet_coco_panoptic_mod


def _rgb_from_id(segment_id: int):
    return np.array(
        [segment_id % 256, (segment_id // 256) % 256, (segment_id // (256 * 256)) % 256],
        dtype=np.uint8,
    )


def _load_dataset_module():
    _install_stub_dependencies()
    module_name = "cvlization.dataset.coco_panoptic_tiny"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_parse_ann_info_filters_crowd_and_maps_categories():
    module = _load_dataset_module()
    ds = module.CocoPanopticTinyDataset()
    ds.cat2label = {101: 0, 202: 1}

    class FakeCoco:
        @staticmethod
        def load_cats(ids):
            return [{"isthing": 1 if ids == 101 else 0}]

    ds.coco = FakeCoco()
    ann = ds._parse_ann_info(
        [
            {"id": 11, "bbox": [10, 20, 5, 6], "area": 30, "category_id": 101},
            {"id": 22, "bbox": [0, 0, 4, 4], "area": 16, "category_id": 202},
            {"id": 33, "bbox": [2, 2, 5, 5], "area": 25, "category_id": 101, "iscrowd": True},
            {"id": 44, "bbox": [1, 1, 0, 3], "area": 0, "category_id": 101},
        ]
    )

    assert ann["bboxes"].shape == (1, 4)
    np.testing.assert_allclose(ann["bboxes"][0], np.array([10, 20, 15, 26], dtype=np.float32))
    np.testing.assert_array_equal(ann["labels"], np.array([0], dtype=np.int64))
    assert ann["bboxes_ignore"].shape == (1, 4)
    np.testing.assert_allclose(ann["bboxes_ignore"][0], np.array([2, 2, 7, 7], dtype=np.float32))

    mask_infos = ann["masks"]
    assert len(mask_infos) == 3
    assert mask_infos[0]["id"] == 11 and bool(mask_infos[0]["is_thing"])
    assert mask_infos[1]["id"] == 22 and not bool(mask_infos[1]["is_thing"])
    assert mask_infos[2]["id"] == 33 and not bool(mask_infos[2]["is_thing"])


def test_parse_pan_label_image_creates_semantic_and_thing_masks():
    module = _load_dataset_module()
    ds = module.CocoPanopticTinyDataset()

    h, w = 2, 3
    pan = np.zeros((h, w, 3), dtype=np.uint8)
    pan[0, 0] = _rgb_from_id(1)
    pan[0, 1] = _rgb_from_id(2)
    pan[1, 1] = _rgb_from_id(1)

    gt_seg, gt_masks = ds._parse_pan_label_image(
        pan,
        [
            {"id": 1, "category": 5, "is_thing": True},
            {"id": 2, "category": 9, "is_thing": False},
        ],
    )

    expected_seg = np.full((h, w), 255, dtype=np.int64)
    expected_seg[0, 0] = 5
    expected_seg[1, 1] = 5
    expected_seg[0, 1] = 9
    np.testing.assert_array_equal(gt_seg, expected_seg)

    assert gt_masks.shape == (1, h, w)
    expected_mask = np.zeros((h, w), dtype=np.uint8)
    expected_mask[0, 0] = 1
    expected_mask[1, 1] = 1
    np.testing.assert_array_equal(gt_masks[0], expected_mask)


def test_parse_pan_label_image_all_stuff_returns_empty_masks():
    module = _load_dataset_module()
    ds = module.CocoPanopticTinyDataset()

    h, w = 2, 3
    pan = np.zeros((h, w, 3), dtype=np.uint8)
    pan[0, 0] = _rgb_from_id(2)
    pan[1, 2] = _rgb_from_id(2)

    gt_seg, gt_masks = ds._parse_pan_label_image(
        pan,
        [
            {"id": 2, "category": 9, "is_thing": False},
        ],
    )

    expected_seg = np.full((h, w), 255, dtype=np.int64)
    expected_seg[0, 0] = 9
    expected_seg[1, 2] = 9
    np.testing.assert_array_equal(gt_seg, expected_seg)
    assert gt_masks.shape == (0, h, w)
    assert gt_masks.dtype == np.uint8


def test_getitem_returns_current_training_contract(tmp_path):
    module = _load_dataset_module()

    dataset_folder = "tiny"
    img_folder = "val2017_subset"
    seg_folder = "val2017_subset_panoptic_masks"
    base_dir = tmp_path / dataset_folder
    (base_dir / img_folder).mkdir(parents=True)
    (base_dir / seg_folder).mkdir(parents=True)

    rgb_img = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb_img[..., 0] = 255
    Image.fromarray(rgb_img).save(base_dir / img_folder / "sample.jpg")

    pan = np.zeros((2, 3, 3), dtype=np.uint8)
    pan[0, 0] = _rgb_from_id(1)
    pan[1, 1] = _rgb_from_id(1)
    pan[0, 1] = _rgb_from_id(2)
    Image.fromarray(pan).save(base_dir / seg_folder / "sample.png")

    ds = module.CocoPanopticTinyDataset(
        data_dir=str(tmp_path),
        dataset_folder=dataset_folder,
        ann_file="annotations/dummy.json",
        channels_first=True,
        label_offset=1,
    )
    ds.annotations = [object()]
    ds.ids = [123]
    ds.cat2label = {101: 0, 202: 1}

    class FakeCoco:
        def __init__(self):
            self.dataset = {"categories": []}

        @staticmethod
        def get_ann_ids(img_ids):
            assert img_ids == [123]
            return [11, 22]

        @staticmethod
        def load_anns(ann_ids):
            assert ann_ids == [11, 22]
            return [
                {"id": 1, "image_id": 123, "bbox": [0, 0, 2, 2], "area": 4, "category_id": 101},
                {"id": 2, "image_id": 123, "bbox": [1, 0, 1, 1], "area": 1, "category_id": 202},
            ]

        @staticmethod
        def load_cats(ids):
            return [{"isthing": 1 if ids == 101 else 0}]

        @staticmethod
        def loadImgs(image_id):
            assert image_id == 123
            return [{"file_name": "sample.jpg"}]

    ds.coco = FakeCoco()

    inputs, targets = ds[0]
    image = inputs[0]
    bboxes, labels, masks, seg_map = targets

    assert image.shape == (3, 2, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0 and image.max() <= 1

    assert bboxes.shape == (1, 4)
    np.testing.assert_allclose(bboxes[0], np.array([0, 0, 2, 2], dtype=np.float32))
    assert labels.shape == (1, 1)
    np.testing.assert_array_equal(labels[:, 0], np.array([1], dtype=np.int64))
    assert masks.shape == (1, 2, 3)
    assert seg_map.shape == (2, 3)
    assert set(np.unique(seg_map).tolist()) == {0, 1, 255}
