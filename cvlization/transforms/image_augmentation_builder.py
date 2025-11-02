from ..specs.transforms.image_augmentation_spec import (
    ImageAugmentationSpec,
    ImageAugmentationProvider,
)


class ImageAugmentationBuilder:
    def __init__(self, spec: ImageAugmentationSpec):
        self.spec = spec

    def run(self):
        provider = self.spec.provider

        if provider == ImageAugmentationProvider.ALBUMENTATIONS:
            from ..transforms.albumentations_transform import AlbumentationsTransform

            albumentations_object = AlbumentationsTransform(config_file_or_dict=self.spec.config)
            return albumentations_object
        elif provider == ImageAugmentationProvider.TORCHVISION:
            from ..torch.transforms.torchvision_transform import TorchvisionTransform

            torchvision_object = TorchvisionTransform(self.spec.config)
            return torchvision_object
        elif provider == ImageAugmentationProvider.KORNIA:
            from ..torch.transforms.kornia_transform import KorniaTransform

            kornia_object = KorniaTransform(self.spec.config)
            return kornia_object
        else:
            raise NotImplementedError(
                f"Image augmentation provider {self.spec.provider} not supported."
            )
