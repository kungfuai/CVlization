import torch
from torch import nn
from typing import Optional, Callable, List
from torchvision.models.video.swin_transformer import PatchEmbed3d


class PatchEmbedOmnivore(nn.Module):
    """Patch Embedding strategy for Omnivore model
    It will use common PatchEmbed3d for image and video,
    for single view depth image it will have separate embedding for the depth channel
    and add the embedding result with the RGB channel
    reference: https://arxiv.org/abs/2201.08377
    Args:
        patch_size (Tuple[int, int, int]): Patch token size. Default: ``(2, 4, 4)``
        embed_dim (int): Number of linear projection output channels. Default: ``96``
        norm_layer (nn.Module, optional): Normalization layer. Default: ``None``
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3d(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        self.depth_patch_embed = PatchEmbed3d(
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C D H W
        # Note: D here represent time
        assert x.ndim == 5
        has_depth = x.shape[1] == 4

        if has_depth:
            x_rgb = self.patch_embed(x[:, :3, ...])
            x_d = self.depth_patch_embed(x[:, 3:, ...])
            x = x_rgb + x_d
        else:
            x = self.patch_embed(x)
        return x


class Omnivore(nn.Module):
    """Omnivore is a model that accept multiple vision modality.
    Omnivore (https://arxiv.org/abs/2201.08377) is a single model that able to do classification
    on images, videos, and single-view 3D data using the same shared parameters of the encoder.
    Args:
        encoder (nn.Module): Instantiated encoder. It generally accept a video backbone.
            The paper use SwinTransformer3d for the encoder.
        heads (Optional[nn.ModuleDict]): Dictionary of multiple heads for each dataset type
    Inputs:
        x (Tensor): 5 Dimensional batched video tensor with format of B C D H W
            where B is batch, C is channel, D is time, H is height, and W is width.
        input_type (str): The dataset type of the input, this will used to choose
            the correct head.
    """

    def __init__(self, encoder: nn.Module, heads: nn.ModuleDict):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, x: torch.Tensor, input_type: str) -> torch.Tensor:
        x = self.encoder(x)
        assert (
            input_type in self.heads
        ), f"Unsupported input_type: {input_type}, please use one of {list(self.heads.keys())}"
        x = self.heads[input_type](x)
        return x
