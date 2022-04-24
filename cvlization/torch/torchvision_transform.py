import json
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
        for t in self.config["transformers"]:
            self.transforms.append(self.get_tf(t))
        self.aug = transforms.Compose(self.transforms)

    def get_tf(self, transform_fn_and_kwargs: dict):
        fn_name = transform_fn_and_kwargs["type"]
        kwargs = transform_fn_and_kwargs.get("kwargs", {})
        return getattr(transforms, fn_name)(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, image, target=None):
        if target is None:
            return self.aug(image)
        return self.aug(image), target
