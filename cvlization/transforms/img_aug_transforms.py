import json
from typing import Union, Callable
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np

from .example_transform import ExampleTransform


TRANSFORM_MENU = {
    "sharpen": iaa.Sharpen,
    "emboss": iaa.Emboss,
    "gaussian_blur": iaa.GaussianBlur,
    "perspective_transform": iaa.PerspectiveTransform,
    "affine": iaa.Affine,
    "color_temp": iaa.ChangeColorTemperature,
    "salt_and_pepper": iaa.SaltAndPepper,
    "flip_lr": iaa.Fliplr,
    "flip_ud": iaa.Flipud,
    "contrast": iaa.LinearContrast,
    "multiply": iaa.Multiply,
    "resize": iaa.Resize,
    "crop": iaa.Crop,
    "dropout": iaa.Dropout,
    "add": iaa.Add,
    "rot90": iaa.geometric.Rot90,  # TODO double check the syntax
}


class ImgAugTransform:
    def __init__(
        self,
        cv_task: str = None,
        config_file_or_dict=None,
        channels_first: bool = True,
        **kwargs,
    ):
        if config_file_or_dict:
            self.aug = self.build_transform_from_config(config_file_or_dict)
        else:
            self.aug = iaa.Sequential(
                [iaa.Sometimes(0.5, iaa.Fliplr()), iaa.Crop(percent=(0, 0.2))]
            )

        if cv_task is None:
            cv_task = self.config.get("cv_task")
        assert isinstance(cv_task, str), f"cv_task must be a str, got {cv_task}"
        assert cv_task.lower() in [
            "classification",
            "detection",
            "semseg",
        ], f"{cv_task} is not a valid cv task"
        self.cv_task = cv_task
        self.channels_first = channels_first

    def transform_tuple(self, x):
        """
        in json, there is no tuple, but imgaug treats lists and tuples differently
        Per the documentation, a tuple creates a uniform distribution between x[0] and x[1]
        I think there is an issue with random generation in detectron2.
        Lists are treated as something to sample the value from, so we can just generate
        a distribution
        """
        if isinstance(x, list):
            if len(x) == 2:
                return tuple(x)
            else:
                return x
        else:
            return x

    def get_tf(self, td):
        prob = td["probability"]
        kwargs = {}
        for k, v in td.get("kwargs", {}).items():
            kwargs[k] = self.transform_tuple(v)
        try:
            return iaa.Sometimes(prob, TRANSFORM_MENU[td["type"]](**kwargs))
        except:
            print(
                f"Error creating augmentation step: {td['type']}, {TRANSFORM_MENU[td['type']]}"
            )
            raise

    def build_transform_from_config(
        self, transform_config_json: Union[str, dict]
    ) -> iaa.Sequential:
        if isinstance(transform_config_json, str):
            with open(transform_config_json, "r") as f:
                config = json.load(f)
        elif isinstance(transform_config_json, dict):
            config = transform_config_json
        else:
            raise TypeError(
                f"transform_config_json must be a str or dict, not {type(transform_config_json)}"
            )

        self.config = config

        t_list = []
        for t in config["steps"]:
            t_list.append(self.get_tf(t))

        seq = iaa.Sequential(t_list)
        return seq

    def _classification_augment(self, image, aug_det):
        image_aug = aug_det(image=image)
        return self._unprepare_after_aug(image_aug=image_aug)

    def _objdet_augment(self, image, target, aug_det):
        bboxes = [BoundingBox(x[0], x[1], x[2], x[3]) for x in target["boxes"]]
        bboxes = BoundingBoxesOnImage(bboxes, image.shape)
        image_aug, bboxes_aug = aug_det(image=image, bounding_boxes=bboxes)
        clipped_boxes = bboxes_aug.remove_out_of_image_fraction(0.75)
        # target["boxes"] = torch.from_numpy(clipped_boxes.to_xyxy_array())
        target["boxes"] = clipped_boxes.to_xyxy_array()
        image_aug = self._unprepare_after_aug(image_aug=image_aug)
        return image_aug, target

    def _semseg_augment(self, image, segmap, aug_det):
        image_aug, segmap = aug_det(image=image, segmentation_maps=segmap)
        image_aug = self._unprepare_after_aug(image_aug=image_aug)
        return image_aug, segmap

    def _prepare_for_aug(self, image):
        if isinstance(image, np.ndarray):
            return image
        return np.moveaxis(image.numpy(), 0, -1)

    def _unprepare_after_aug(self, image_aug):
        # return torch.from_numpy(np.moveaxis(image_aug.copy(), -1, 0))
        # torch.from_numpy to be applied outside of this function.
        return np.moveaxis(image_aug.copy(), -1, 0)

    def transform_image_and_targets(
        self,
        image,
        bboxes=None,
        mask=None,
        keypoints=None,
    ) -> dict:
        assert image is not None, "image is None"
        # TODO: add support for channels_last
        if self.channels_first:
            # imgaug assumes channels last, so we need to transpose the image.
            assert image.shape[0] <= 3, "image is not channels_first"
            image = np.transpose(image, (1, 2, 0))
        aug_det = self.aug.to_deterministic()
        input_dict = dict(
            image=image,
            bounding_boxes=bboxes,
            segmentation_maps=mask,
            keypoints=keypoints,
        )
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        if "bounding_boxes" in input_dict:
            bboxes = [
                BoundingBox(x[0], x[1], x[2], x[3])
                for x in input_dict["bounding_boxes"]
            ]
            bboxes = BoundingBoxesOnImage(bboxes, image.shape)
            input_dict["bounding_boxes"] = bboxes
        if "segmentation_maps" in input_dict:
            input_segmap = input_dict["segmentation_maps"]
            input_segmap = np.squeeze(input_segmap)
            assert (
                input_segmap.ndim == 2
            ), f"segmentation_maps must be 2D, got shape {input_segmap.shape}"
            segmap = SegmentationMapsOnImage(input_segmap, image.shape)
            input_dict["segmentation_maps"] = segmap

        keys = [k for k in input_dict]
        output: tuple = aug_det(**input_dict)
        assert len(output) == len(
            keys
        ), f"output length {len(output)} != keys length {len(keys)}"
        key_mapping = self.get_key_mapping()
        dict_np_output = {key_mapping[k]: v for k, v in zip(keys, output)}
        for k, v in dict_np_output.items():
            if hasattr(v, "get_arr"):
                dict_np_output[k] = v.get_arr()  # draw()
            elif k == "image":
                # The output of imgaug is channels_last. So if `channels_first`
                # is set to True, we need to transpose the image.
                if self.channels_first:
                    dict_np_output[k] = np.transpose(v, (2, 0, 1))
        # print("***** transformed!!")
        # for k, v in dict_np_output.items():
        #     if isinstance(v, np.ndarray):
        #         print(f"{k}: {v.shape}")
        #     elif isinstance(v, list):
        #         for i in v[:2]:
        #             print(f"{k}: {i.shape}")
        # assert False
        return dict_np_output

    def get_key_mapping(self) -> dict:
        return {
            "image": "image",
            "bounding_boxes": "bboxes",
            "segmentation_maps": "mask",
            "keypoints": "keypoints",
        }

    def transform(self, image, target=None, segmap=None):
        """
        Expects a 3 dimensional image with dim_0 = channels
        """
        assert target == None or type(target) in [dict, list, tuple]

        image = self._prepare_for_aug(image=image)
        aug_det = self.aug.to_deterministic()

        if self.cv_task.lower() == "classification":
            image_aug = self._classification_augment(image=image, aug_det=aug_det)
            return image_aug
        elif self.cv_task.lower() == "detection":
            assert type(target) == dict
            assert "boxes" and "labels" in target.keys()
            image_aug, target = self._objdet_augment(
                image=image, target=target, aug_det=aug_det
            )
            return image_aug, target
        elif self.cv_task.lower() == "semseg":
            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            image_aug, segmap = self._semseg_augment(
                image=image, segmap=segmap, aug_det=aug_det
            )
            return image_aug, segmap
        else:
            raise Exception(
                "The type of augmentation neede could not be determined. Please ensure that target = None \
                or that target is a dictionary containing bboxes and labels"
            )

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
