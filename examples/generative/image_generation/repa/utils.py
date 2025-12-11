"""
Utility functions for REPA training and inference.
Adapted from https://github.com/sihyun-yu/REPA
"""

import os
import math
import warnings
import torch
import timm
from torchvision.datasets.utils import download_url


# Pretrained model download
pretrained_models = {'last.pt'}


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = 'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    """
    Load pretrained visual encoders for REPA alignment.
    Supports: dinov2-vit-{s,b,l,g}
    """
    assert resolution in (256, 512), "Resolution must be 256 or 512"

    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []

    for enc_name in enc_names:
        parts = enc_name.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid encoder name format: {enc_name}. Expected format: type-arch-size (e.g., dinov2-vit-b)")

        encoder_type, architecture, model_config = parts

        # Currently only support DINOv2 for simplicity
        if encoder_type not in ('dinov2', 'dinov2reg'):
            raise NotImplementedError(
                f"Encoder type '{encoder_type}' not supported. Use dinov2-vit-{{s,b,l,g}}"
            )

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        # Load DINOv2
        if 'reg' in encoder_type:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')

        # Remove classification head
        if hasattr(encoder, 'head'):
            del encoder.head

        # Resample positional embeddings for resolution
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(device)
        encoder.eval()

        encoders.append(encoder)

    return encoders, encoder_types, architectures


def load_legacy_checkpoints(state_dict, encoder_depth):
    """Convert legacy checkpoint format to new format."""
    new_state_dict = dict()
    for key, value in state_dict.items():
        if 'decoder_blocks' in key:
            parts = key.split('.')
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = 'blocks'
            parts[1] = str(new_idx)
            new_key = '.'.join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
