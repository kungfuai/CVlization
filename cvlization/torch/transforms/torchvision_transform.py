import json
from PIL import Image
from typing import Union
from torchvision import transforms


class TorchvisionTransform:
    def __init__(self, config: Union[str, dict]):
        if isinstance(config, str):
            self.config = json.load(open(config))
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be a str or dict, not {type(config)}")

        self.transforms = []
        for t in self.config["steps"]:
            self.transforms.append(self.get_tf(t))
        self.aug = transforms.Compose(self.transforms)

    def get_tf(self, transform_fn_and_kwargs: dict):
        fn_name = transform_fn_and_kwargs["type"]
        kwargs = transform_fn_and_kwargs.get("kwargs", {})
        return getattr(transforms, fn_name)(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform_image_and_targets(self, image, **kwargs):
        if image.shape[0] <= 3 and len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        return {
            "image": self.aug(image),
            **kwargs,
        }

    def transform(self, example: Union[tuple, list, dict]) -> Union[tuple, list, dict]:
        if isinstance(example, tuple) or isinstance(example, list):
            assert len(example) >= 2
            image = example[0]
            image = self.aug(image)
            if isinstance(example, tuple):
                return (image,) + example[1:]
            elif isinstance(example, list):
                return [image] + example[1:]
        elif isinstance(example, dict):
            image = example["image"]
            target = example.get("label")
            return {
                "image": self.aug(image),
                "label": target,
            }
        elif isinstance(example, Image.Image):
            return self.aug(example)
        else:
            raise TypeError(
                f"example must be a tuple, list, or dict, not {type(example)}"
            )
