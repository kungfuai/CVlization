"""
LaMa (Large Mask Inpainting) with Temporal Attention and Mask Prediction

Based on: "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
https://github.com/advimman/lama

Key features:
- Fast Fourier Convolutions (FFC) for global receptive field
- Temporal attention for video consistency
- Mask prediction head for artifact detection
- Support for loading pretrained LaMa weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

# Reuse temporal attention from existing model
from model import TemporalAttention, LayerNorm2d


class FourierUnit(nn.Module):
    """
    Fourier Unit: applies convolution in frequency domain.

    This gives the network a global receptive field without
    the computational cost of large kernels or transformers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 1,
        fft_norm: str = "ortho",
    ):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        # Convolution in frequency domain (on real and imaginary parts)
        self.conv = nn.Conv2d(
            in_channels * 2,  # real + imaginary
            out_channels * 2,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        # FFT: spatial -> frequency domain
        # rfft2 returns complex tensor, we'll work with real representation
        fft_x = torch.fft.rfft2(x, norm=self.fft_norm)

        # Stack real and imaginary parts along channel dimension
        fft_x = torch.stack([fft_x.real, fft_x.imag], dim=-1)
        fft_x = fft_x.permute(0, 1, 4, 2, 3).contiguous()
        fft_x = fft_x.view(batch, -1, fft_x.shape[-2], fft_x.shape[-1])

        # Convolution in frequency domain
        fft_x = self.conv(fft_x)
        fft_x = self.bn(fft_x)
        fft_x = self.relu(fft_x)

        # Reshape back to complex
        fft_x = fft_x.view(batch, -1, 2, fft_x.shape[-2], fft_x.shape[-1])
        fft_x = fft_x.permute(0, 1, 3, 4, 2).contiguous()
        fft_x = torch.complex(fft_x[..., 0], fft_x[..., 1])

        # Inverse FFT: frequency -> spatial domain
        output = torch.fft.irfft2(fft_x, s=x.shape[-2:], norm=self.fft_norm)

        return output


class SpectralTransform(nn.Module):
    """
    Spectral Transform block: combines local and global (Fourier) processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
    ):
        super().__init__()
        self.enable_lfu = enable_lfu

        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        if enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            output = output + self.lfu(x)

        output = self.conv2(output)
        return output


class FFC(nn.Module):
    """
    Fast Fourier Convolution: combines local and global branches.

    The local branch uses standard convolutions.
    The global branch uses Fourier transforms for global receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        ratio_gin: float = 0.5,  # ratio of global input channels
        ratio_gout: float = 0.5,  # ratio of global output channels
        groups: int = 1,
        enable_lfu: bool = True,
    ):
        super().__init__()

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl

        # Local to local
        if in_cl > 0 and out_cl > 0:
            self.conv_l2l = nn.Conv2d(
                in_cl, out_cl, kernel_size, stride, padding, groups=groups, bias=False
            )

        # Local to global
        if in_cl > 0 and out_cg > 0:
            self.conv_l2g = nn.Conv2d(
                in_cl, out_cg, kernel_size, stride, padding, groups=groups, bias=False
            )

        # Global to local
        if in_cg > 0 and out_cl > 0:
            self.conv_g2l = nn.Conv2d(
                in_cg, out_cl, kernel_size, stride, padding, groups=groups, bias=False
            )

        # Global to global (spectral)
        if in_cg > 0 and out_cg > 0:
            self.conv_g2g = SpectralTransform(
                in_cg, out_cg, stride, groups, enable_lfu
            )

        self.bn_l = nn.BatchNorm2d(out_cl) if out_cl > 0 else None
        self.bn_g = nn.BatchNorm2d(out_cg) if out_cg > 0 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into local and global parts
        if self.in_cg > 0:
            x_l, x_g = x[:, :self.in_cl], x[:, self.in_cl:]
        else:
            x_l, x_g = x, None

        # Local branch
        out_l = 0
        if self.in_cl > 0 and self.out_cl > 0:
            out_l = self.conv_l2l(x_l)
        if self.in_cg > 0 and self.out_cl > 0:
            out_l = out_l + self.conv_g2l(x_g)

        # Global branch
        out_g = 0
        if self.in_cl > 0 and self.out_cg > 0:
            out_g = self.conv_l2g(x_l)
        if self.in_cg > 0 and self.out_cg > 0:
            out_g = out_g + self.conv_g2g(x_g)

        # Apply normalization and activation
        if self.out_cl > 0:
            out_l = self.act(self.bn_l(out_l))
        if self.out_cg > 0:
            out_g = self.act(self.bn_g(out_g))

        # Concatenate local and global
        if self.out_cl > 0 and self.out_cg > 0:
            return torch.cat([out_l, out_g], dim=1)
        elif self.out_cl > 0:
            return out_l
        else:
            return out_g


class FFCResBlock(nn.Module):
    """
    Residual block using FFC.
    """

    def __init__(
        self,
        channels: int,
        ratio_gin: float = 0.5,
        ratio_gout: float = 0.5,
    ):
        super().__init__()

        self.ffc1 = FFC(channels, channels, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.ffc2 = FFC(channels, channels, ratio_gin=ratio_gout, ratio_gout=ratio_gout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ffc1(x)
        out = self.ffc2(out)
        return out + residual


class LamaEncoder(nn.Module):
    """
    LaMa encoder: standard convolutions for downsampling.
    """

    def __init__(
        self,
        in_channels: int = 4,  # RGB + mask
        base_channels: int = 64,
        num_downs: int = 3,
    ):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Downsampling layers
        self.downs = nn.ModuleList()
        ch = base_channels
        for i in range(num_downs):
            self.downs.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ))
            ch *= 2

        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []

        x = self.in_conv(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        return x, skips[:-1]  # Don't include last as skip


class LamaDecoder(nn.Module):
    """
    LaMa decoder: upsampling with skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        num_ups: int = 3,
    ):
        super().__init__()

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = in_channels
        for i in range(num_ups):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ))
            ch //= 2

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_channels, kernel_size=7),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(skips):
                # Add skip connection
                skip = skips[-(i + 1)]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = x + skip

        return self.out_conv(x)


class LamaWithMask(nn.Module):
    """
    LaMa architecture with temporal attention and mask prediction.

    Architecture:
        Input (RGB) -> Encoder -> FFC ResBlocks -> Temporal Attention -> Decoder -> Output
                                                                    \\-> Mask Head -> Mask

    Output is composited: output = input * (1 - mask) + inpainted * mask
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_downs: int = 3,
        num_ffc_blocks: int = 9,
        ratio_g: float = 0.5,  # ratio of global channels in FFC
        use_temporal_attention: bool = True,
        num_frames: int = 5,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_temporal_attention = use_temporal_attention

        # Encoder (no mask input - we predict it)
        self.encoder = LamaEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downs=num_downs,
        )

        bottleneck_ch = self.encoder.out_channels

        # FFC ResBlocks (the core of LaMa)
        self.ffc_blocks = nn.Sequential(*[
            FFCResBlock(bottleneck_ch, ratio_gin=ratio_g, ratio_gout=ratio_g)
            for _ in range(num_ffc_blocks)
        ])

        # Temporal attention
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(
                bottleneck_ch,
                num_heads=attention_heads,
                num_frames=num_frames,
            )

        # Decoder for inpainting
        self.decoder = LamaDecoder(
            in_channels=bottleneck_ch,
            out_channels=out_channels,
            num_ups=num_downs,
        )

        # Mask prediction head (from bottleneck features)
        self.mask_decoder = LamaDecoder(
            in_channels=bottleneck_ch,
            out_channels=1,
            num_ups=num_downs,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_inpainted: bool = False,
        inject_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C, H, W] video frames or [B, C, H, W] single frame
            return_inpainted: If True, also return raw inpainted image
            inject_mask: Optional [B, T, 1, H, W] or [B, 1, H, W] mask to use instead of predicted.
                         Useful for ablation studies, interactive editing, or two-stage pipelines.

        Returns:
            output: Composited result (input * (1-mask) + inpainted * mask)
            pred_mask: Predicted artifact mask (always returned for loss computation)
            inpainted (optional): Raw inpainted image before compositing
        """
        # Handle different input formats
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            is_video = True
            x_input = x.view(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1
            is_video = False
            x_input = x

        # Encode
        feat, skips = self.encoder(x_input)

        # FFC blocks (global receptive field)
        feat = self.ffc_blocks(feat)

        # Temporal attention
        if self.use_temporal_attention and is_video:
            feat = feat.view(B, T, -1, feat.shape[-2], feat.shape[-1])
            feat = self.temporal_attn(feat)
            feat = feat.view(B * T, -1, feat.shape[-2], feat.shape[-1])

        # Decode to get inpainted image and mask
        inpainted = self.decoder(feat, skips)
        pred_mask = self.mask_decoder(feat, skips)

        # Use injected mask if provided, otherwise use predicted mask
        if inject_mask is not None:
            # Reshape inject_mask to [B*T, 1, H, W] if needed
            if inject_mask.dim() == 5:
                inject_mask = inject_mask.view(B * T, 1, H, W)
            composite_mask = inject_mask
        else:
            # Detach predicted mask so recon_loss only trains inpaint path
            composite_mask = pred_mask.detach()

        # Explicit composite: preserve clean regions exactly
        out = x_input * (1 - composite_mask) + inpainted * composite_mask

        # Reshape for video output
        if is_video:
            out = out.view(B, T, self.out_channels, H, W)
            pred_mask = pred_mask.view(B, T, 1, H, W)
            inpainted = inpainted.view(B, T, self.out_channels, H, W)

        if return_inpainted:
            return out, pred_mask, inpainted
        return out, pred_mask

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained LaMa weights.

        Args:
            checkpoint_path: Path to LaMa checkpoint
            strict: If False, ignore missing/unexpected keys

        Note: LaMa checkpoints have different structure, this maps weights appropriately.
        Mask decoder is initialized randomly (not in original LaMa).
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'generator' in state_dict:
            state_dict = state_dict['generator']

        # Map LaMa keys to our model keys
        # LaMa uses: model.encoder, model.decoder, model.ffc_blocks
        # Our model uses: encoder, decoder, ffc_blocks
        mapped_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            new_key = key.replace('model.', '')

            # Skip mask-related keys (we have our own mask decoder)
            if 'mask' in new_key.lower():
                continue

            mapped_state_dict[new_key] = value

        # Load with strict=False to handle missing mask decoder weights
        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=strict)

        if missing:
            print(f"Missing keys (expected, mask decoder is new): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        return missing, unexpected

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Smaller variant for faster experiments
class LamaLite(LamaWithMask):
    """
    Smaller LaMa variant with fewer FFC blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        use_temporal_attention: bool = True,
        num_frames: int = 5,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=48,  # Smaller
            num_downs=3,
            num_ffc_blocks=4,  # Fewer blocks
            use_temporal_attention=use_temporal_attention,
            num_frames=num_frames,
        )


# Test
if __name__ == "__main__":
    print("Testing LaMa components...")

    # Test FourierUnit
    fu = FourierUnit(64, 64)
    x = torch.randn(2, 64, 32, 32)
    y = fu(x)
    print(f"FourierUnit: {x.shape} -> {y.shape}")

    # Test FFC
    ffc = FFC(64, 64, ratio_gin=0.5, ratio_gout=0.5)
    x = torch.randn(2, 64, 32, 32)
    y = ffc(x)
    print(f"FFC: {x.shape} -> {y.shape}")

    # Test FFCResBlock
    block = FFCResBlock(64)
    x = torch.randn(2, 64, 32, 32)
    y = block(x)
    print(f"FFCResBlock: {x.shape} -> {y.shape}")

    # Test full model (single frame)
    print("\nTesting LamaWithMask (single frame)...")
    model = LamaWithMask(use_temporal_attention=False)
    x = torch.randn(2, 3, 256, 256)
    out, mask = model(x)
    print(f"Single frame: {x.shape} -> output {out.shape}, mask {mask.shape}")
    print(f"Parameters: {model.count_parameters():,}")

    # Test full model (video)
    print("\nTesting LamaWithMask (video)...")
    model = LamaWithMask(use_temporal_attention=True, num_frames=5)
    x = torch.randn(2, 5, 3, 256, 256)
    out, mask = model(x)
    print(f"Video: {x.shape} -> output {out.shape}, mask {mask.shape}")
    print(f"Parameters: {model.count_parameters():,}")

    # Test with return_inpainted
    out, mask, inpainted = model(x, return_inpainted=True)
    print(f"With inpainted: output {out.shape}, mask {mask.shape}, inpainted {inpainted.shape}")

    # Test LamaLite
    print("\nTesting LamaLite...")
    model_lite = LamaLite(use_temporal_attention=True, num_frames=5)
    x = torch.randn(2, 5, 3, 256, 256)
    out, mask = model_lite(x)
    print(f"LamaLite: {x.shape} -> output {out.shape}, mask {mask.shape}")
    print(f"Parameters: {model_lite.count_parameters():,}")

    print("\nAll tests passed!")
