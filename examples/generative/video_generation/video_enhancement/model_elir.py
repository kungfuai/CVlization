"""
ELIR: Efficient Latent Image Restoration with Flow Matching

Adapted from: https://github.com/KAIST-VML/ELIR (Apache-2.0 License)

This is a frame-by-frame implementation for video artifact removal.
Key components:
- TAESD: Tiny AutoEncoder for latent space (8x downsample)
- LUnet: Lightweight U-Net with time embeddings for flow model
- Flow matching: K-step ODE integration in latent space

Training:
- Flow matching loss: train velocity prediction at random timesteps
- Mask loss: auxiliary task for artifact detection
- Composite output: output = input*(1-mask) + restored*mask

Architecture: enc -> mmse -> K flow steps (fmir) -> dec -> composite with mask
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model import NAFBlock, DownBlock, UpBlock  # Reuse proven architecture


# =============================================================================
# Sinusoidal Position Embedding (for flow time step)
# =============================================================================

def sinusoidal_pos_emb(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """
    Generate sinusoidal position embedding for time step.

    Args:
        t: Time step tensor [B] or scalar
        dim: Embedding dimension (must be even)
        scale: Scale factor for time

    Returns:
        Embedding tensor [B, dim] or [1, dim]
    """
    assert dim % 2 == 0, "Dimension must be even"

    if t.dim() == 0:
        t = t.unsqueeze(0)

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb


# =============================================================================
# Timestep Embedding MLP
# =============================================================================

class TimestepEmbedding(nn.Module):
    """MLP to project timestep embedding."""

    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_emb_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        return self.linear2(x)


# =============================================================================
# TAESD: Tiny AutoEncoder for Stable Diffusion
# =============================================================================

def taesd_conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    """3x3 conv with padding=1 for TAESD."""
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class TAESDBlock(nn.Module):
    """Residual block for TAESD."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            taesd_conv(n_in, n_out),
            nn.ReLU(),
            taesd_conv(n_out, n_out),
            nn.ReLU(),
            taesd_conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


class TAESDEncoder(nn.Module):
    """TAESD Encoder: 3 -> latent_channels with 8x downsample."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 16, hidden_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            taesd_conv(in_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            # 2x downsample
            taesd_conv(hidden_channels, hidden_channels, stride=2, bias=False),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            # 4x downsample
            taesd_conv(hidden_channels, hidden_channels, stride=2, bias=False),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            # 8x downsample
            taesd_conv(hidden_channels, hidden_channels, stride=2, bias=False),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            # Project to latent
            taesd_conv(hidden_channels, latent_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TAESDDecoder(nn.Module):
    """TAESD Decoder: latent_channels -> 3 with 8x upsample."""

    def __init__(self, out_channels: int = 3, latent_channels: int = 16, hidden_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            taesd_conv(latent_channels, hidden_channels),
            nn.ReLU(),
            # 2x upsample
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            taesd_conv(hidden_channels, hidden_channels, bias=False),
            # 4x upsample
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            taesd_conv(hidden_channels, hidden_channels, bias=False),
            # 8x upsample
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            TAESDBlock(hidden_channels, hidden_channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            taesd_conv(hidden_channels, hidden_channels, bias=False),
            # Output
            TAESDBlock(hidden_channels, hidden_channels),
            taesd_conv(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp latent range like original TAESD
        x = torch.tanh(x / 3) * 3
        return self.layers(x)


# =============================================================================
# LUnet Components: Lightweight U-Net for Flow Model
# =============================================================================

class LUnetBlock(nn.Module):
    """Basic block with GroupNorm + SiLU + Conv."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        # Ensure groups doesn't exceed channels
        groups = min(groups, in_channels)
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LUnetResBlock(nn.Module):
    """ResNet block with time embedding injection."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = LUnetBlock(in_channels, out_channels, groups)
        self.block2 = LUnetBlock(out_channels, out_channels, groups)
        # Named conv2d to match original ELIR checkpoint
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add time embedding
        h = h + self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.conv2d(x)


class LUnetDownsample(nn.Module):
    """Downsample with strided conv."""

    def __init__(self, in_channels: int, out_channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.conv(x)
        else:
            return self.avgpool(self.conv(x))


class LUnetUpsample(nn.Module):
    """Upsample with transposed conv or interpolation."""

    def __init__(self, in_channels: int, out_channels: int, use_convtr: bool = True):
        super().__init__()
        self.use_convtr = use_convtr
        if use_convtr:
            self.convtr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtr:
            return self.convtr(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            return self.conv(x)


class LUnet(nn.Module):
    """
    Lightweight U-Net for flow model with time embedding.

    Args:
        in_channels: Input latent channels
        hid_channels: Base hidden channels
        out_channels: Output latent channels
        ch_mult: Channel multipliers for each stage
        n_mid_blocks: Number of middle blocks
        t_emb_dim: Time embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 16,
        hid_channels: int = 128,
        out_channels: int = 16,
        ch_mult: List[int] = [1, 2, 1, 2],
        n_mid_blocks: int = 3,
        t_emb_dim: int = 160,
        use_rescale_conv: bool = True,
    ):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        time_dim_out = 4 * t_emb_dim

        self.time_mlp = TimestepEmbedding(t_emb_dim, time_dim_out)
        self.first_proj = nn.Conv2d(in_channels, hid_channels, kernel_size=1)

        self.down_blocks = nn.ModuleList()
        self.mid_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Down blocks
        chs = hid_channels
        for mult in ch_mult:
            resnet = LUnetResBlock(chs, chs, time_dim_out)
            if mult != 1:
                downsample = LUnetDownsample(chs, mult * chs, use_conv=use_rescale_conv)
                chs = mult * chs
            else:
                downsample = nn.Identity()
            self.down_blocks.append(nn.ModuleList([resnet, downsample]))

        # Mid blocks
        for _ in range(n_mid_blocks):
            self.mid_blocks.append(LUnetResBlock(chs, chs, time_dim_out))

        # Up blocks
        for mult in ch_mult[::-1]:
            if mult != 1:
                upsample = LUnetUpsample(chs, chs // mult, use_convtr=use_rescale_conv)
                chs = chs // mult
            else:
                upsample = nn.Identity()
            resnet = LUnetResBlock(2 * chs, chs, time_dim_out)
            self.up_blocks.append(nn.ModuleList([upsample, resnet]))

        self.final_block = LUnetBlock(chs, chs)
        self.final_proj = nn.Conv2d(chs, out_channels, kernel_size=1)

    def forward(self, xt: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: Noisy latent [B, C, H, W]
            t_emb: Time embedding [B, t_emb_dim]

        Returns:
            Velocity prediction [B, C, H, W]
        """
        emb = self.time_mlp(t_emb)
        x = self.first_proj(xt)

        # Down
        skip_connect = []
        for resnet, downsample in self.down_blocks:
            x = resnet(x, emb)
            skip_connect.append(x)
            x = downsample(x)

        # Mid
        for resnet in self.mid_blocks:
            x = resnet(x, emb)

        # Up
        for upsample, resnet in self.up_blocks:
            x = upsample(x)
            x = torch.cat([x, skip_connect.pop()], dim=1)
            x = resnet(x, emb)

        x = self.final_block(x)
        x = self.final_proj(x)
        return x


# =============================================================================
# RRDBNet: MMSE Network (Original ELIR architecture)
# Residual-in-Residual Dense Block Network for initial latent estimation
# =============================================================================

class RRDBBlock2D(nn.Module):
    """Basic conv block for RRDB."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, activation: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel, padding=1),
            nn.SiLU() if activation else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 convolutions.
    Dense connections: each conv receives all previous features.
    """

    def __init__(self, c_in: int = 16, c_hid: int = 96):
        super().__init__()
        self.conv1 = RRDBBlock2D(c_in, c_hid, 3, True)
        self.conv2 = RRDBBlock2D(c_in + c_hid, c_hid, 3, True)
        self.conv3 = RRDBBlock2D(c_in + 2 * c_hid, c_hid, 3, True)
        self.conv4 = RRDBBlock2D(c_in + 3 * c_hid, c_hid, 3, True)
        self.conv5 = RRDBBlock2D(c_in + 4 * c_hid, c_in, 3, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.
    Contains 3 ResidualDenseBlock_5C with residual scaling.
    """

    def __init__(self, c_inout: int = 16, c_hid: int = 96):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(c_inout, c_hid)
        self.RDB2 = ResidualDenseBlock_5C(c_inout, c_hid)
        self.RDB3 = ResidualDenseBlock_5C(c_inout, c_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet: Latent Reconstruction Module (MMSE estimator).

    Original ELIR architecture for initial latent estimate.
    Uses dense connections for better gradient flow and feature reuse.

    Args:
        c_inout: Input/output channels (latent channels, default 16)
        c_hid: Hidden channels for dense blocks (default 96, matches pretrained)
        n_rrdb: Number of RRDB blocks (default 3, matches pretrained BFR)
    """

    def __init__(self, c_inout: int = 16, c_hid: int = 96, n_rrdb: int = 3):
        super().__init__()
        layers = []
        for _ in range(n_rrdb):
            layers.append(RRDB(c_inout, c_hid))
        self.lrm = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lrm(x)


# =============================================================================
# Mask Head (for composite output)
# =============================================================================

class MaskHead(nn.Module):
    """Mask prediction using NAFBlock encoder (same architecture as TemporalNAFUNet).

    Uses lightweight NAFBlocks with LayerNorm, gating, and channel attention
    for robust mask prediction. This matches the proven approach from
    ExplicitCompositeNet and TemporalNAFUNet.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, num_blocks: int = 2):
        super().__init__()
        # Initial projection (like ExplicitCompositeNet)
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        # NAFBlocks for feature extraction
        self.blocks = nn.Sequential(*[NAFBlock(hidden_channels) for _ in range(num_blocks)])

        # Mask head (same as ExplicitCompositeNet)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.in_conv(x)
        feat = self.blocks(feat)
        return self.mask_conv(feat)


class MaskUNet(nn.Module):
    """
    NAFNet-style U-Net for mask prediction (same architecture as ExplicitCompositeNet).

    This provides a proven, powerful mask predictor that uses full encoder-decoder
    with skip connections, matching the architecture that works well in
    ExplicitCompositeNet and TemporalNAFUNet.

    Architecture:
        in_conv -> 3 DownBlocks -> bottleneck -> 3 UpBlocks -> mask_head

    Args:
        in_channels: Input image channels (default 3 for RGB)
        encoder_channels: Channel progression [32, 64, 128, 256]
        num_blocks: NAFBlocks per encoder/decoder stage
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: list = None,
        num_blocks: int = 2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, encoder_channels[0], 3, padding=1)

        # Encoder: 3 DownBlocks
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                DownBlock(encoder_channels[i], encoder_channels[i + 1], num_blocks)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(encoder_channels[-1])
                                          for _ in range(num_blocks)])

        # Decoder: 3 UpBlocks
        self.decoders = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                UpBlock(decoder_channels[i], decoder_channels[i + 1], num_blocks)
            )

        # Mask head (same as ExplicitCompositeNet)
        self.mask_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0] // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W]

        Returns:
            mask: Predicted mask [B, 1, H, W]
        """
        # Initial conv
        feat = self.in_conv(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            feat, skip = encoder(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            feat = decoder(feat, skip)

        # Mask head
        return self.mask_head(feat)


class MaskDecoder(nn.Module):
    """
    Mask decoder using encoder latent features with skip connections (Option B).

    Takes 8x downsampled latent features from TAESD encoder and upsamples to
    full resolution mask. Uses skip connections from encoder intermediate
    features for better multi-scale information.

    Uses LayerNorm (GroupNorm with groups=1) to maintain variance through the
    network. This prevents activation collapse when encoder features have low
    variance.

    Architecture:
        latent (16ch, H/8) + skip_4x (64ch, H/4) + skip_2x (64ch, H/2) + skip_1x (64ch, H)
        -> upsample 8x -> mask (1ch, H)
    """

    def __init__(self, latent_channels: int = 16, hidden_channels: int = 64, encoder_channels: int = 64):
        super().__init__()
        # Process latent features with LayerNorm (GroupNorm g=1)
        self.conv_in = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.norm_in = nn.GroupNorm(1, hidden_channels)  # LayerNorm equivalent

        # Skip connection projections (encoder uses hidden_channels=64)
        # These project encoder features to match decoder channels
        self.skip_proj_4x = nn.Conv2d(encoder_channels, hidden_channels, 1)
        self.skip_proj_2x = nn.Conv2d(encoder_channels, hidden_channels // 2, 1)
        self.skip_proj_1x = nn.Conv2d(encoder_channels, hidden_channels // 4, 1)

        # Upsample path (8x total: 3 stages of 2x)
        # Using LayerNorm after each conv to maintain activation variance
        # After skip connection fusion, we use conv to blend features
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),  # concat skip
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(1, hidden_channels // 2),
            nn.ReLU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),  # concat skip
            nn.GroupNorm(1, hidden_channels // 2),
            nn.ReLU(),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, 3, padding=1),
            nn.GroupNorm(1, hidden_channels // 4),
            nn.ReLU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, 3, padding=1),  # concat skip
            nn.GroupNorm(1, hidden_channels // 4),
            nn.ReLU(),
        )

        # Output with learnable temperature for sharper predictions
        self.conv_out = nn.Conv2d(hidden_channels // 4, 1, 1)
        # Initialize temperature to 1.0, can learn to increase for sharper masks
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            z: Latent features [B, latent_channels, H/8, W/8]
            skips: Optional list of encoder skip features [skip_4x, skip_2x, skip_1x]
                   Each has shape [B, encoder_channels, H/4, H/2, H] respectively
        Returns:
            mask: Predicted mask [B, 1, H, W]
        """
        x = F.relu(self.norm_in(self.conv_in(z)))

        # Up1: H/8 -> H/4
        x = self.up1(x)
        if skips is not None and len(skips) >= 1 and skips[0] is not None:
            skip = self.skip_proj_4x(skips[0])
            x = self.fuse1(torch.cat([x, skip], dim=1))

        # Up2: H/4 -> H/2
        x = self.up2(x)
        if skips is not None and len(skips) >= 2 and skips[1] is not None:
            skip = self.skip_proj_2x(skips[1])
            x = self.fuse2(torch.cat([x, skip], dim=1))

        # Up3: H/2 -> H
        x = self.up3(x)
        if skips is not None and len(skips) >= 3 and skips[2] is not None:
            skip = self.skip_proj_1x(skips[2])
            x = self.fuse3(torch.cat([x, skip], dim=1))

        # Apply temperature scaling before sigmoid for sharper predictions
        logits = self.conv_out(x)
        return torch.sigmoid(logits * self.temperature.clamp(min=0.1))


# =============================================================================
# ElirWithMask: Main Model
# =============================================================================

class ElirWithMask(nn.Module):
    """
    ELIR model adapted for video artifact removal with mask prediction.

    Training uses proper flow matching loss:
    1. Encode both degraded and clean to latent space
    2. Sample random timestep t in [0, 1]
    3. Interpolate: z_t = (1-t)*z_degraded + t*z_clean
    4. Predict velocity: v_pred = flow(z_t, t)
    5. Target velocity: v_target = z_clean - z_degraded
    6. Flow loss: MSE(v_pred, v_target)

    Inference uses K-step ODE:
    1. Encode degraded to latent
    2. MMSE initial estimate
    3. K Euler steps with flow model
    4. Decode to pixel space
    5. Composite with predicted mask

    Args:
        in_channels: Input image channels
        out_channels: Output image channels
        latent_channels: Latent space channels
        hidden_channels: Hidden channels in encoder/decoder
        flow_hidden_channels: Hidden channels in flow model
        k_steps: Number of flow matching steps (NFE) for inference
        t_emb_dim: Time embedding dimension
        sigma_s: Initial noise scale for MMSE
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        flow_hidden_channels: int = 128,
        k_steps: int = 3,
        t_emb_dim: int = 160,
        sigma_s: float = 0.1,
        use_temporal_attention: bool = False,  # Placeholder for future
        num_frames: int = 5,
        attention_heads: int = 4,
        use_mask_decoder: bool = False,  # Option B: use encoder features for mask
        use_mask_unet: bool = False,  # Option C: use full NAFNet UNet for mask (like ExplicitCompositeNet)
        detach_mask_features: bool = True,  # Detach encoder features for mask (False for training from scratch)
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.k_steps = k_steps
        self.t_emb_dim = t_emb_dim
        self.sigma_s = sigma_s
        self.dt = 1.0 / k_steps
        self.use_mask_decoder = use_mask_decoder
        self.use_mask_unet = use_mask_unet
        self.detach_mask_features = detach_mask_features

        # Encoder: image -> latent (TAESD with fixed 64 hidden channels like original)
        self.encoder = TAESDEncoder(in_channels, latent_channels, hidden_channels=64)

        # MMSE: initial latent estimate using RRDBNet (original ELIR architecture)
        # c_hid=96 for dense blocks, n_rrdb=3 (matches pretrained BFR weights)
        self.mmse = RRDBNet(c_inout=latent_channels, c_hid=96, n_rrdb=3)

        # Flow model: velocity prediction
        self.flow = LUnet(
            in_channels=latent_channels,
            hid_channels=flow_hidden_channels,
            out_channels=latent_channels,
            ch_mult=[1, 2, 1, 2],
            n_mid_blocks=3,
            t_emb_dim=t_emb_dim,
        )

        # Decoder: latent -> image (TAESD with fixed 64 hidden channels like original)
        self.decoder = TAESDDecoder(out_channels, latent_channels, hidden_channels=64)

        # Mask prediction: three options
        # Option A (default): Simple MaskHead on raw pixels
        # Option B (use_mask_decoder=True): MaskDecoder on encoder latent features with skip connections
        # Option C (use_mask_unet=True): Full NAFNet UNet (same as ExplicitCompositeNet)
        if use_mask_unet:
            # Option C: Full NAFNet-style UNet for mask prediction
            # This matches ExplicitCompositeNet's proven architecture
            self.mask_unet = MaskUNet(in_channels, encoder_channels=[32, 64, 128, 256], num_blocks=2)
            self.mask_head = None
            self.mask_decoder = None
            self._encoder_skips = None
            print("Using MaskUNet (Option C): full NAFNet encoder-decoder for mask (like ExplicitCompositeNet)")
        elif use_mask_decoder:
            self.mask_decoder = MaskDecoder(latent_channels, hidden_channels=64, encoder_channels=64)
            self.mask_head = None
            self.mask_unet = None
            print("Using MaskDecoder (Option B): mask from encoder features with skip connections")
            # Setup hooks to capture encoder intermediate features for skip connections
            self._encoder_skips = {}
            self._setup_encoder_hooks()
        else:
            self.mask_head = MaskHead(in_channels, hidden_channels=32)
            self.mask_decoder = None
            self.mask_unet = None
            self._encoder_skips = None

        # Placeholder for temporal attention
        self.use_temporal_attention = use_temporal_attention

    def _setup_encoder_hooks(self):
        """Register forward hooks to capture encoder intermediate features."""
        # TAESD encoder layer indices for skip connections:
        # Layer 1: after first block, before 2x downsample (H resolution)
        # Layer 5: after 2x stage blocks, before 4x downsample (H/2 resolution)
        # Layer 9: after 4x stage blocks, before 8x downsample (H/4 resolution)

        def make_hook(name):
            def hook(module, input, output):
                self._encoder_skips[name] = output
            return hook

        # Register hooks on specific layers
        self.encoder.layers[1].register_forward_hook(make_hook('skip_1x'))   # H resolution
        self.encoder.layers[5].register_forward_hook(make_hook('skip_2x'))   # H/2 resolution
        self.encoder.layers[9].register_forward_hook(make_hook('skip_4x'))   # H/4 resolution

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)

    def predict_velocity(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity at timestep t."""
        t_emb = sinusoidal_pos_emb(t, self.t_emb_dim)
        if t_emb.shape[0] == 1 and z_t.shape[0] > 1:
            t_emb = t_emb.expand(z_t.shape[0], -1)
        return self.flow(z_t, t_emb)

    def compute_flow_loss(
        self,
        degraded: torch.Tensor,
        clean: torch.Tensor,
        sigma_min: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute flow matching loss for training (matches original ELIR formulation).

        Original ELIR uses:
        1. Encode degraded -> MMSE estimate -> add noise
        2. Interpolate with sigma_min: Xt = (1 - (1-sigma_min)*t) * X_mmse_noisy + t * X_hq
        3. Target: u = X_hq - (1-sigma_min) * X_mmse_noisy

        Args:
            degraded: Degraded images [B, C, H, W] or [B*T, C, H, W]
            clean: Clean images, same shape
            sigma_min: Minimum sigma for flow matching (default 1e-5)

        Returns:
            Flow matching loss (MSE between predicted and target velocity)
        """
        B = degraded.shape[0]

        # Encode clean image (target) - no grad for stability
        with torch.no_grad():
            z_clean = self.encode(clean)

        # Encode degraded and compute MMSE estimate
        z_degraded = self.encode(degraded)
        z_mmse = self.mmse(z_degraded)

        # Add noise to MMSE output (detached to prevent gradient through MMSE for flow)
        eps = torch.randn_like(z_mmse)
        z_mmse_noisy = z_mmse.detach() + self.sigma_s * eps

        # Sample random timestep t in [0, 1]
        t = torch.rand(B, 1, 1, 1, device=degraded.device, dtype=degraded.dtype)

        # Interpolate with sigma_min (original ELIR formulation)
        # Xt = (1 - (1-sigma_min)*t) * X_mmse_noisy + t * X_hq
        z_t = (1 - (1 - sigma_min) * t) * z_mmse_noisy + t * z_clean

        # Predict velocity
        v_pred = self.predict_velocity(z_t, t.squeeze())

        # Target velocity (original ELIR formulation)
        # u = X_hq - (1-sigma_min) * X_mmse_noisy
        v_target = z_clean - (1 - sigma_min) * z_mmse_noisy

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def flow_loop(self, z0: torch.Tensor) -> torch.Tensor:
        """K-step ODE integration for inference."""
        z = z0
        t = 0.0
        for _ in range(self.k_steps):
            t_tensor = torch.tensor([t], device=z.device)
            velocity = self.predict_velocity(z, t_tensor)
            z = z + self.dt * velocity
            t += self.dt
        return z

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        """
        Restore image using ELIR flow matching (no mask).

        Args:
            x: Degraded image [B, C, H, W]

        Returns:
            Restored image [B, C, H, W]
        """
        # Encode
        z = self.encode(x)
        return self._restore_from_latent(z)

    def _restore_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Restore image from pre-computed latent (used when sharing encoder).

        Args:
            z: Encoded latent [B, latent_channels, H/8, W/8]

        Returns:
            Restored image [B, C, H, W]
        """
        # MMSE initial estimate with optional noise
        if self.training:
            noise = self.sigma_s * torch.randn_like(z)
        else:
            noise = 0.0
        z0 = self.mmse(z) + noise

        # Flow loop
        z_restored = self.flow_loop(z0)

        # Decode
        restored = self.decode(z_restored)
        restored = torch.sigmoid(restored)  # Ensure [0, 1]

        return restored

    def forward(
        self,
        x: torch.Tensor,
        clean: Optional[torch.Tensor] = None,
        return_inpainted: bool = False,
        inject_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]:
        """
        Forward pass with optional flow matching loss computation.

        Args:
            x: Degraded input [B, T, C, H, W] video or [B, C, H, W] image
            clean: Clean target (required for training with flow loss)
            return_inpainted: If True, also return raw restored output
            inject_mask: Optional mask to use instead of predicted

        Returns:
            If clean is provided (training mode):
                Dict with 'output', 'pred_mask', 'restored', 'flow_loss'
            Otherwise (inference mode):
                (output, pred_mask) or (output, pred_mask, restored)
        """
        # Handle video input
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            is_video = True
            x_input = x.view(B * T, C, H, W)
            if clean is not None:
                clean_input = clean.view(B * T, C, H, W)
            else:
                clean_input = None
        else:
            B, C, H, W = x.shape
            T = 1
            is_video = False
            x_input = x
            clean_input = clean

        # Predict mask - three options:
        # Option A: Simple MaskHead on raw pixels
        # Option B: MaskDecoder on encoder latent features (detached)
        # Option C: Full MaskUNet (same as ExplicitCompositeNet)
        if self.use_mask_unet:
            # Option C: Full NAFNet-style UNet for mask prediction
            pred_mask = self.mask_unet(x_input)
            # Restore image using ELIR
            restored = self.restore(x_input)
        elif self.use_mask_decoder:
            # Option B: Use encoder features for mask prediction with skip connections
            # Clear previous skip features
            self._encoder_skips.clear()
            # Encode first - hooks will capture intermediate features
            z_input = self.encode(x_input)
            # Collect skip features captured by hooks [skip_4x, skip_2x, skip_1x]
            skips = [
                self._encoder_skips.get('skip_4x'),  # H/4 resolution
                self._encoder_skips.get('skip_2x'),  # H/2 resolution
                self._encoder_skips.get('skip_1x'),  # H resolution
            ]
            # Detach skip features if needed
            if self.detach_mask_features:
                skips = [s.detach() if s is not None else None for s in skips]
                z_for_mask = z_input.detach()
            else:
                z_for_mask = z_input
            # Predict mask from encoder features with skip connections
            pred_mask = self.mask_decoder(z_for_mask, skips=skips)
            # Clear skip features to free memory
            self._encoder_skips.clear()
            # Restore using pre-computed latent
            restored = self._restore_from_latent(z_input)
        else:
            # Option A: Simple mask head on raw pixels
            pred_mask = self.mask_head(x_input)
            # Restore image using ELIR
            restored = self.restore(x_input)

        # Use injected mask if provided, otherwise use predicted mask
        if inject_mask is not None:
            if inject_mask.dim() == 5:
                inject_mask = inject_mask.view(B * T, 1, H, W)
            composite_mask = inject_mask
        else:
            composite_mask = pred_mask.detach()

        # Composite: preserve clean regions, use restored in artifact regions
        # Detach restored so flow model is trained ONLY by flow_loss, not recon_loss
        # (This is specific to ELIR which uses flow matching - other models don't detach)
        restored_for_composite = restored.detach() if self.training else restored
        out = x_input * (1 - composite_mask) + restored_for_composite * composite_mask

        # Compute flow loss if clean is provided (training mode)
        flow_loss = None
        if clean_input is not None:
            flow_loss = self.compute_flow_loss(x_input, clean_input)

        # Reshape for video output
        if is_video:
            out = out.view(B, T, self.out_channels, H, W)
            pred_mask = pred_mask.view(B, T, 1, H, W)
            restored = restored.view(B, T, self.out_channels, H, W)

        # Return format depends on mode
        if clean is not None:
            # Training mode: return dict with all components
            return {
                'output': out,
                'pred_mask': pred_mask,
                'restored': restored,
                'flow_loss': flow_loss,
            }
        elif return_inpainted:
            return out, pred_mask, restored
        else:
            return out, pred_mask

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_elir_weights(self, path: str, strict: bool = False):
        """
        Load pretrained ELIR weights.

        Supports two checkpoint formats:
        1. .ckpt files with separate state dicts (state_dict_enc, state_dict_mmse, etc.)
        2. .pth files with combined state dict (enc., mmse., fmir., dec. prefixes)

        Args:
            path: Path to ELIR checkpoint
            strict: If False, ignore missing/unexpected keys (mask_head, etc.)
        """
        # weights_only=False needed for some checkpoint formats
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle .ckpt format with separate state dicts
        if path.endswith(".ckpt") or "state_dict_enc" in checkpoint:
            print("Loading ELIR weights from .ckpt format (separate state dicts)")
            # Load each component separately
            if "state_dict_enc" in checkpoint:
                missing_enc, _ = self.encoder.load_state_dict(
                    checkpoint["state_dict_enc"], strict=False
                )
                print(f"  Encoder: loaded (missing: {len(missing_enc)})")
            if "state_dict_mmse" in checkpoint:
                missing_mmse, _ = self.mmse.load_state_dict(
                    checkpoint["state_dict_mmse"], strict=False
                )
                print(f"  MMSE (RRDBNet): loaded (missing: {len(missing_mmse)})")
            if "state_dict_fmir" in checkpoint:
                missing_flow, _ = self.flow.load_state_dict(
                    checkpoint["state_dict_fmir"], strict=False
                )
                print(f"  Flow (LUnet): loaded (missing: {len(missing_flow)})")
            if "state_dict_dec" in checkpoint:
                missing_dec, _ = self.decoder.load_state_dict(
                    checkpoint["state_dict_dec"], strict=False
                )
                print(f"  Decoder: loaded (missing: {len(missing_dec)})")
            print("  Mask head: randomly initialized (not in pretrained)")
            return

        # Handle .pth format with combined state dict
        state_dict = checkpoint
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Map ELIR keys to our keys
        key_map = {
            "enc.": "encoder.",
            "mmse.": "mmse.",
            "fmir.": "flow.",
            "dec.": "decoder.",
        }

        mapped_state = {}
        for k, v in state_dict.items():
            new_key = k
            for old, new in key_map.items():
                if k.startswith(old):
                    new_key = new + k[len(old):]
                    break
            mapped_state[new_key] = v

        # Load with strict=False to allow mask_head to be randomly initialized
        missing, unexpected = self.load_state_dict(mapped_state, strict=False)

        if not strict:
            print(f"Loaded ELIR weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            if missing:
                # Only show mask_head as expected missing
                mask_head_missing = [k for k in missing if k.startswith("mask_head")]
                other_missing = [k for k in missing if not k.startswith("mask_head")]
                if mask_head_missing:
                    print(f"  Mask head: randomly initialized ({len(mask_head_missing)} params)")
                if other_missing:
                    print(f"  WARNING: Other missing keys: {other_missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected[:5]}...")
        else:
            if missing or unexpected:
                raise RuntimeError(f"Missing: {missing}, Unexpected: {unexpected}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test model
    model = ElirWithMask(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        hidden_channels=64,
        flow_hidden_channels=128,
        k_steps=3,
    )

    print(f"ElirWithMask parameters: {model.count_parameters():,}")

    # Test inference mode (no clean)
    x = torch.randn(2, 3, 128, 128).clamp(0, 1)
    out, mask = model(x)
    print(f"Inference: {x.shape} -> output: {out.shape}, mask: {mask.shape}")

    # Test training mode (with clean)
    clean = torch.randn(2, 3, 128, 128).clamp(0, 1)
    result = model(x, clean=clean)
    print(f"Training: output: {result['output'].shape}, mask: {result['pred_mask'].shape}, "
          f"flow_loss: {result['flow_loss'].item():.4f}")

    # Test with video
    x = torch.randn(2, 5, 3, 128, 128).clamp(0, 1)
    clean = torch.randn(2, 5, 3, 128, 128).clamp(0, 1)
    result = model(x, clean=clean)
    print(f"Video training: output: {result['output'].shape}, flow_loss: {result['flow_loss'].item():.4f}")

    # Test component sizes
    print("\nComponent parameters:")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  MMSE: {sum(p.numel() for p in model.mmse.parameters()):,}")
    print(f"  Flow (LUnet): {sum(p.numel() for p in model.flow.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"  Mask head: {sum(p.numel() for p in model.mask_head.parameters()):,}")

    # Test MaskUNet option (Option C)
    print("\n--- Testing MaskUNet (Option C) ---")
    model_unet = ElirWithMask(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        hidden_channels=64,
        flow_hidden_channels=128,
        k_steps=3,
        use_mask_unet=True,
    )
    print(f"ElirWithMask (MaskUNet) parameters: {model_unet.count_parameters():,}")
    print(f"  MaskUNet: {sum(p.numel() for p in model_unet.mask_unet.parameters()):,}")

    # Test inference
    x = torch.randn(2, 3, 128, 128).clamp(0, 1)
    out, mask = model_unet(x)
    print(f"Inference with MaskUNet: {x.shape} -> output: {out.shape}, mask: {mask.shape}")
