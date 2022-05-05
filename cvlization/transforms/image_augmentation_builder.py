from ..specs.transforms.image_augmentation_spec import (
    ImageAugmentationSpec,
    ImageAugmentationProvider,
)


class ImageAugmentationBuilder:
    def __init__(self, spec: ImageAugmentationSpec):
        self.spec = spec

    def run(self):
        if self.spec.provider == ImageAugmentationProvider.IMGAUG:
            from ..transforms.img_aug_transforms import ImgAugTransform

            img_aug_object = ImgAugTransform(self.spec.config)
            return img_aug_object
        elif self.spec.provider == ImageAugmentationProvider.TORCHVISION:
            # TODO
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
