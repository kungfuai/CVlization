import json
from typing import Union
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np


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


class ImgAugTransforms(object):
    def __init__(self, cv_task, config_file=None):
        assert cv_task.lower() in ["classification", "detection", "semseg"]
        self.cv_task = cv_task

        if config_file:
            self.aug = self.build_transform_from_config(config_file)
        else:
            self.aug = iaa.Sequential(
                [iaa.Sometimes(0.5, iaa.Fliplr()), iaa.Crop(percent=(0, 0.2))]
            )

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
        return iaa.Sometimes(prob, TRANSFORM_MENU[td["type"]](**kwargs))

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

        t_list = []
        for t in config["transformers"]:
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
        return np.moveaxis(image.numpy(), 0, -1)

    def _unprepare_after_aug(self, image_aug):
        # return torch.from_numpy(np.moveaxis(image_aug.copy(), -1, 0))
        # torch.from_numpy to be applied outside of this function.
        return np.moveaxis(image_aug.copy(), -1, 0)

    def transform(self, image, target=None, segmap=None):
        """
        Expects a 3 dimensional image with dim_0 = channels
        """
        assert target == None or type(target) == dict

        image = self._prepare_for_aug(image=image)
        aug_det = self.aug.to_deterministic()

        if self.cv_task.lower() == "classification":
            # if isinstance(target, type(None)) and isinstance(segmap, type(None)):
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
