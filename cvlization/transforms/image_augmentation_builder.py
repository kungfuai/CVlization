from ..specs.transforms.image_augmentation import (
    ImageAugmentation,
    ImageAugmentationProvider,
)


class ImageAugmentationBuilder:
    def __init__(self, spec: ImageAugmentation):
        self.spec = spec

    def run(self):
        if self.spec.provider == ImageAugmentationProvider.IMGAUG:
            from ..transforms.img_aug_transforms import ImgAugTransform

            img_aug_object = ImgAugTransform(self.spec.config)
            return img_aug_object
        elif self.spec.provider == ImageAugmentationProvider.TORCHVISION:
            # TODO
            from ..torch.torchvision_transform import TorchvisionTransform

            torchvision_object = TorchvisionTransform(self.spec.config)
            return torchvision_object
        else:
            raise NotImplementedError(
                f"Image augmentation provider {self.spec.provider} not supported."
            )
