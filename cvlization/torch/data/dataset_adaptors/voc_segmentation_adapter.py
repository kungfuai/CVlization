import torch
import torchvision


class VOCSegmentationAdapter:
    target_transform = torchvision.transforms.ToTensor()
