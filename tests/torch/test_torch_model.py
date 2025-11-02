import importlib.util

from cvlization.torch.encoder.torch_image_backbone import image_backbone_names
from cvlization.torch.encoder.torch_image_backbone import BUILTIN_MODELS

def test_can_get_image_backbone_names():
    names = image_backbone_names()
    if importlib.util.find_spec("timm"):
        assert len(names) > 300, f"Expected large model list when timm is installed."
    else:
        # Without timm we still expect builtin backbones to be present.
        for builtin in BUILTIN_MODELS:
            assert builtin in names, f"Missing builtin backbone {builtin}"
        assert len(names) >= len(BUILTIN_MODELS)
