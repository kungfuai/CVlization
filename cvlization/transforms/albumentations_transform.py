import json
from typing import Union
import numpy as np
import albumentations as A


class AlbumentationsTransform:
    """
    Albumentations-based image augmentation transform for CVlization.
    Compatible with the ImgAugTransform interface.
    """

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
            # Default augmentation
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomCrop(width=256, height=256, p=1.0),
                ]
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

    def build_transform_from_config(
        self, transform_config_json: Union[str, dict]
    ) -> A.Compose:
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

        # Map from config transform names to albumentations transforms
        transform_map = {
            "resize": lambda kwargs: A.Resize(
                height=kwargs["size"]["height"],
                width=kwargs["size"]["width"]
            ),
            "flip_lr": lambda kwargs: A.HorizontalFlip(),
            "flip_ud": lambda kwargs: A.VerticalFlip(),
            "rotate": lambda kwargs: A.Rotate(**kwargs),
            "crop": lambda kwargs: A.Crop(**kwargs),
            "blur": lambda kwargs: A.Blur(**kwargs),
            "gaussian_blur": lambda kwargs: A.GaussianBlur(**kwargs),
            "brightness_contrast": lambda kwargs: A.RandomBrightnessContrast(**kwargs),
            "hue_saturation": lambda kwargs: A.HueSaturationValue(**kwargs),
            "normalize": lambda kwargs: A.Normalize(**kwargs),
        }

        for step in config["steps"]:
            prob = step["probability"]
            transform_type = step["type"]
            kwargs = step.get("kwargs", {})

            if transform_type in transform_map:
                transform = transform_map[transform_type](kwargs)
                # Apply probability if < 1.0
                if prob < 1.0:
                    t_list.append(A.OneOf([transform], p=prob))
                else:
                    t_list.append(transform)
            else:
                print(f"Warning: Transform '{transform_type}' not supported in albumentations")

        # Create Compose with appropriate bbox/mask parameters based on cv_task
        if self.config.get("cv_task", "").lower() == "detection":
            compose = A.Compose(
                t_list,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
            )
        elif self.config.get("cv_task", "").lower() == "semseg":
            compose = A.Compose(t_list)
        else:
            compose = A.Compose(t_list)

        return compose

    def transform_image_and_targets(
        self,
        image,
        bboxes=None,
        mask=None,
        keypoints=None,
    ) -> dict:
        """
        Main transformation method compatible with ImgAugTransform interface.

        Args:
            image: numpy array (C, H, W) if channels_first else (H, W, C)
            bboxes: list of bounding boxes in pascal_voc format [x_min, y_min, x_max, y_max]
            mask: segmentation mask (C, H, W) or (H, W, C)
            keypoints: list of keypoints

        Returns:
            dict with keys: 'image', 'bboxes', 'mask', 'keypoints'
        """
        assert image is not None, "image is None"

        # Albumentations expects channels_last (H, W, C)
        if self.channels_first and image.ndim == 3:
            assert image.shape[0] <= 3, f"image is not channels_first, shape: {image.shape}"
            image = np.transpose(image, (1, 2, 0))

        # Prepare albumentations input
        albu_input = {"image": image}

        if bboxes is not None and len(bboxes) > 0:
            # Albumentations expects list of bboxes
            albu_input["bboxes"] = bboxes
            # Create dummy labels if not provided (required by albumentations)
            albu_input["labels"] = list(range(len(bboxes)))

        if mask is not None:
            # Handle mask: albumentations expects (H, W) for semantic segmentation
            if mask.ndim == 3:
                if self.channels_first:
                    # (C, H, W) -> (H, W, C)
                    mask = np.transpose(mask, (1, 2, 0))
                # Squeeze if single channel
                if mask.shape[2] == 1:
                    mask = np.squeeze(mask, axis=2)
            albu_input["mask"] = mask

        if keypoints is not None and len(keypoints) > 0:
            albu_input["keypoints"] = keypoints

        # Apply augmentation
        try:
            augmented = self.aug(**albu_input)
        except Exception as e:
            print(f"Error during albumentations transform: {e}")
            print(f"Input shapes - image: {image.shape}, mask: {mask.shape if mask is not None else None}")
            raise

        # Prepare output dict
        output_dict = {}

        # Handle image output
        aug_image = augmented["image"]
        if self.channels_first and aug_image.ndim == 3:
            # Convert back to channels_first (H, W, C) -> (C, H, W)
            aug_image = np.transpose(aug_image, (2, 0, 1))
        output_dict["image"] = aug_image

        # Handle bboxes output
        if "bboxes" in augmented:
            output_dict["bboxes"] = np.array(augmented["bboxes"])

        # Handle mask output
        if "mask" in augmented:
            aug_mask = augmented["mask"]
            # For semantic segmentation, keep mask as 2D (H, W)
            # For instance segmentation, it would be 3D (H, W, C) or (C, H, W)
            if self.cv_task.lower() == "semseg":
                # Ensure mask is 2D for semantic segmentation
                if aug_mask.ndim == 3:
                    # Squeeze out single channel dimension if present
                    if aug_mask.shape[-1] == 1:
                        aug_mask = np.squeeze(aug_mask, axis=-1)
                    elif aug_mask.shape[0] == 1:
                        aug_mask = np.squeeze(aug_mask, axis=0)
            else:
                # For other tasks, convert mask back to channels_first if needed
                if aug_mask.ndim == 2:
                    # Add channel dimension
                    aug_mask = np.expand_dims(aug_mask, axis=0)
                elif aug_mask.ndim == 3 and self.channels_first:
                    # (H, W, C) -> (C, H, W)
                    aug_mask = np.transpose(aug_mask, (2, 0, 1))
            output_dict["mask"] = aug_mask

        # Handle keypoints output
        if "keypoints" in augmented:
            output_dict["keypoints"] = augmented["keypoints"]

        return output_dict

    def get_key_mapping(self) -> dict:
        """Return key mapping for compatibility with ImgAugTransform interface."""
        return {
            "image": "image",
            "bounding_boxes": "bboxes",
            "segmentation_maps": "mask",
            "keypoints": "keypoints",
        }

    def transform(self, image, target=None, segmap=None):
        """
        Transform method for backward compatibility with ImgAugTransform interface.
        Expects a 3 dimensional image with dim_0 = channels (if channels_first=True).
        """
        if self.cv_task.lower() == "classification":
            result = self.transform_image_and_targets(image=image)
            return result["image"]

        elif self.cv_task.lower() == "detection":
            assert isinstance(target, dict), "target must be a dict for detection"
            assert "boxes" in target and "labels" in target, \
                "target must have 'boxes' and 'labels' keys"

            result = self.transform_image_and_targets(
                image=image,
                bboxes=target["boxes"]
            )
            target["boxes"] = result.get("bboxes", target["boxes"])
            return result["image"], target

        elif self.cv_task.lower() == "semseg":
            # For semantic segmentation
            result = self.transform_image_and_targets(
                image=image,
                mask=segmap
            )
            aug_mask = result.get("mask", segmap)
            return result["image"], aug_mask

        else:
            raise Exception(
                f"Unknown cv_task: {self.cv_task}. "
                "Please ensure cv_task is 'classification', 'detection', or 'semseg'."
            )

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
