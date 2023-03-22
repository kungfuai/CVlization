from torch import nn


class ImageEnergy(nn.Module):
    """
    An energy model for images.

    The output is a scalar energy value, indicating
    the negative log probability of the input image,
    up to a constant.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # TODO: allow adding additional layers
        self.fc = nn.Linear(backbone.fc.in_features, 1)

    def forward(self, image):
        embedding = self.backbone(image)
        energy = self.fc(embedding)
        return energy