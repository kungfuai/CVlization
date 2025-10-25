from ..specs.transforms.image_augmentation_spec import (
    ImageAugmentationSpec,
    ImageAugmentationProvider,
)


class ImageAugmentationBuilder:
    def __init__(self, spec: ImageAugmentationSpec):
        self.spec = spec

    def run(self):
        if self.spec.provider == ImageAugmentationProvider.IMGAUG:
            import warnings
            warnings.warn(
                "IMGAUG provider is deprecated due to NumPy 2.x compatibility issues. "
                "Please use ALBUMENTATIONS provider instead for better compatibility with modern dependencies.",
                DeprecationWarning,
                stacklevel=2
            )
            from ..transforms.img_aug_transforms import ImgAugTransform

            img_aug_object = ImgAugTransform(config_file_or_dict=self.spec.config)
            return img_aug_object
        elif self.spec.provider == ImageAugmentationProvider.ALBUMENTATIONS:
            from ..transforms.albumentations_transform import AlbumentationsTransform

            albumentations_object = AlbumentationsTransform(config_file_or_dict=self.spec.config)
            return albumentations_object
        elif self.spec.provider == ImageAugmentationProvider.TORCHVISION:
            from ..torch.transforms.torchvision_transform import TorchvisionTransform

            torchvision_object = TorchvisionTransform(self.spec.config)
            return torchvision_object
        elif self.spec.provider == ImageAugmentationProvider.KORNIA:
            from ..torch.transforms.kornia_transform import KorniaTransform

            kornia_object = KorniaTransform(self.spec.config)
            return kornia_object
        else:
            raise NotImplementedError(
                f"Image augmentation provider {self.spec.provider} not supported."
            )
