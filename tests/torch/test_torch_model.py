from cvlization.torch.encoder.torch_image_backbone import image_backbone_names


def test_can_get_image_backbone_names():
    names = image_backbone_names()
    assert len(names) > 300, f"Got image backbone names: {names}."
