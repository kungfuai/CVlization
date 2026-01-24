"""
Video Artifact Removal Model

Architecture: 2D U-Net with optional temporal attention
Optimized for Apple M4 inference (avoids 3D convolutions)

Supports:
- Direct prediction (predict clean frame)
- Residual prediction (predict residual = clean - degraded, then add to input)
- Optional mask prediction head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps"""
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    """Simple gating mechanism (splits channels and multiplies)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet-style block (Simple Baselines for Image Restoration)
    Efficient and well-suited for Apple Silicon
    """
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * expansion, 1)
        self.conv2 = nn.Conv2d(channels * expansion, channels * expansion, 3, 
                               padding=1, groups=channels * expansion)
        self.gate = SimpleGate()
        self.conv3 = nn.Conv2d(channels * expansion // 2, channels, 1)
        
        # Simple channel attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * expansion, 1)
        self.conv5 = nn.Conv2d(channels * expansion, channels * expansion, 3,
                               padding=1, groups=channels * expansion)
        self.gate2 = SimpleGate()
        self.conv6 = nn.Conv2d(channels * expansion // 2, channels, 1)
        
        # Learnable scaling
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.gate(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        
        # Second block
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.gate2(y)
        y = self.conv6(y)
        x = x + y * self.gamma
        
        return x


class TemporalAttention(nn.Module):
    """
    Lightweight temporal attention across frames.
    Processes spatial features and attends across time dimension.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_frames: int = 5,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = LayerNorm2d(channels)
        
        # Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Learnable temporal position embedding
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*T, C, H, W] or [B, T, C, H, W]
        Returns:
            [B*T, C, H, W] or [B, T, C, H, W]
        """
        # Handle input shape
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            reshape_back = True
        else:
            BT, C, H, W = x.shape
            T = min(BT, self.temporal_pos.shape[1])
            B = BT // T
            reshape_back = False
        
        # Pool spatial dimensions for efficiency
        x_pooled = F.adaptive_avg_pool2d(x, (4, 4))  # [B*T, C, 4, 4]
        x_pooled = self.norm(x_pooled)
        
        # QKV
        qkv = self.qkv(x_pooled)  # [B*T, 3*C, 4, 4]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim, 16)
        qkv = qkv.permute(2, 0, 3, 4, 1, 5)  # [3, B, heads, head_dim, T, 16]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Add temporal position embedding
        pos = self.temporal_pos[:, :T].view(1, 1, self.channels, T, 1)
        pos = pos.view(1, self.num_heads, self.head_dim, T, 1)
        q = q + pos
        k = k + pos
        
        # Reshape for attention: [B, heads, T*16, head_dim]
        q = q.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, T * 16, self.head_dim)
        k = k.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, T * 16, self.head_dim)
        v = v.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, T * 16, self.head_dim)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v  # [B, heads, T*16, head_dim]
        out = out.view(B, self.num_heads, T, 16, self.head_dim)
        out = out.permute(0, 2, 1, 4, 3).reshape(B * T, C, 4, 4)
        
        out = self.proj(out)
        
        # Upsample back to original spatial size and add residual
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        result = x + out * 0.1  # Small residual scaling
        
        if reshape_back:
            result = result.view(B, T, C, H, W)

        return result


class MaskGuidedTemporalAttention(nn.Module):
    """
    Temporal attention with mask-based gating.

    Boosts attention to artifact regions by scaling features based on mask
    before computing attention weights.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_frames: int = 5):
        super().__init__()
        self.base_attn = TemporalAttention(channels, num_heads, num_frames)
        # Learnable scale for mask influence
        self.mask_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] features
            mask: [B*T, 1, H', W'] mask at bottleneck resolution

        Returns:
            [B, T, C, H, W] attended features
        """
        B, T, C, H, W = x.shape

        # Reshape mask to [B, T, 1, H', W'] and upsample to feature resolution
        mask_bt = mask.view(B, T, 1, mask.shape[-2], mask.shape[-1])
        if mask_bt.shape[-2:] != (H, W):
            mask_bt = F.interpolate(
                mask_bt.view(B * T, 1, mask.shape[-2], mask.shape[-1]),
                size=(H, W), mode='bilinear', align_corners=False
            ).view(B, T, 1, H, W)

        # Boost features in artifact regions before attention
        # This makes attention "pay more attention" to artifacts
        x_boosted = x * (1 + self.mask_scale * mask_bt)

        # Run base attention on boosted features
        return self.base_attn(x_boosted)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        # All blocks operate at in_ch, downconv changes channels
        self.blocks = nn.Sequential(*[NAFBlock(in_ch) for _ in range(num_blocks)])
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.blocks = nn.Sequential(*[NAFBlock(out_ch) for _ in range(num_blocks)])
        # Skip connection fusion
        self.skip_conv = nn.Conv2d(out_ch * 2, out_ch, 1)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class MaskGuidedModulation(nn.Module):
    """
    Modulate features based on predicted mask.

    Applies: feat = feat * (1 + scale * mask) + bias * mask
    where scale and bias are learned per-channel parameters.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B*T, 1, H, W] or [B, T, 1, H, W]
        # feat: [B*T, C, H, W]
        if mask.dim() == 5:
            B, T, _, H, W = mask.shape
            mask = mask.view(B * T, 1, H, W)
        return feat * (1 + self.scale * mask) + self.bias * mask


class TemporalNAFUNet(nn.Module):
    """
    U-Net with NAFNet-style blocks and optional temporal attention.

    Architecture:
    - Encoder: NAFBlocks with downsampling
    - Bottleneck: NAFBlocks + optional temporal attention
    - Decoder: NAFBlocks with upsampling and skip connections
    - Heads: RGB output + optional mask prediction (multi-task)

    Mask guidance modes:
    - "none": No mask guidance (default)
    - "modulation": Use predicted mask to modulate decoder features

    Designed for video enhancement, optimized for Apple M4 (no 3D convolutions).
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_channels: List[int] = [32, 64, 128, 256],
        num_blocks: int = 2,
        use_temporal_attention: bool = True,
        num_frames: int = 5,
        attention_heads: int = 4,
        residual_learning: bool = False,
        predict_mask: bool = False,
        mask_guidance: str = "none",  # "none" or "modulation"
    ):
        """
        Args:
            in_channels: Input channels (default 3 for RGB)
            out_channels: Output channels (default 3 for RGB)
            encoder_channels: Channel progression for encoder
            num_blocks: NAFBlocks per encoder/decoder stage
            use_temporal_attention: Enable temporal attention at bottleneck
            num_frames: Number of frames for temporal attention
            attention_heads: Number of attention heads
            residual_learning: If True, predict residual and add to input
            predict_mask: If True, also output a mask showing where artifacts were
            mask_guidance: How to use predicted mask ("none" or "modulation")
        """
        super().__init__()

        self.use_temporal_attention = use_temporal_attention
        self.num_frames = num_frames
        self.residual_learning = residual_learning
        self.predict_mask = predict_mask
        self.mask_guidance = mask_guidance
        self.out_channels = out_channels

        # Validate mask_guidance
        valid_guidance = ("none", "modulation", "concat", "residual", "skip_gate", "attn_gate")
        if mask_guidance not in valid_guidance:
            raise ValueError(f"mask_guidance must be one of {valid_guidance}, got '{mask_guidance}'")
        if mask_guidance != "none" and not predict_mask:
            raise ValueError("mask_guidance requires predict_mask=True")

        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, encoder_channels[0], 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                DownBlock(encoder_channels[i], encoder_channels[i + 1], num_blocks)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(encoder_channels[-1])
                                          for _ in range(num_blocks)])

        # Optional temporal attention at bottleneck
        if use_temporal_attention:
            if mask_guidance == "attn_gate":
                self.temporal_attn = MaskGuidedTemporalAttention(
                    encoder_channels[-1],
                    num_heads=attention_heads,
                    num_frames=num_frames
                )
                # Early mask prediction for attention gating
                self.early_mask_conv = nn.Sequential(
                    nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 4, 1),
                    nn.GELU(),
                    nn.Conv2d(encoder_channels[-1] // 4, 1, 1),
                    nn.Sigmoid(),
                )
            else:
                self.temporal_attn = TemporalAttention(
                    encoder_channels[-1],
                    num_heads=attention_heads,
                    num_frames=num_frames
                )

        # Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                UpBlock(decoder_channels[i], decoder_channels[i + 1], num_blocks)
            )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], out_channels, 1),
        )

        # Optional mask prediction head
        if predict_mask:
            self.mask_conv = nn.Sequential(
                nn.Conv2d(encoder_channels[0], encoder_channels[0] // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(encoder_channels[0] // 2, 1, 1),
                nn.Sigmoid(),
            )

        # Optional mask-guided modulation
        if mask_guidance == "modulation":
            self.mask_modulation = MaskGuidedModulation(encoder_channels[0])
        elif mask_guidance == "concat":
            # Concat mask to features, need adjusted output conv
            self.out_conv = nn.Sequential(
                nn.Conv2d(encoder_channels[0] + 1, encoder_channels[0], 3, padding=1),
                nn.GELU(),
                nn.Conv2d(encoder_channels[0], out_channels, 1),
            )
        elif mask_guidance == "skip_gate":
            # Early mask prediction from bottleneck for skip gating
            self.early_mask_conv = nn.Sequential(
                nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 4, 1),
                nn.GELU(),
                nn.Conv2d(encoder_channels[-1] // 4, 1, 1),
                nn.Sigmoid(),
            )
            # Learnable gate strength (how much to suppress corrupted features)
            self.skip_gate_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W] video frames or [B, C, H, W] single frame

        Returns:
            If predict_mask=False: Same shape as input
            If predict_mask=True: Tuple of (output, mask) where mask is [B, T, 1, H, W] or [B, 1, H, W]
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

        # Initial conv
        feat = self.in_conv(x_input)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            feat, skip = encoder(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Early mask prediction for attn_gate (before temporal attention)
        early_mask = None
        if self.mask_guidance == "attn_gate":
            early_mask = self.early_mask_conv(feat)  # [B*T, 1, H', W']

        # Temporal attention
        if self.use_temporal_attention and is_video:
            feat = feat.view(B, T, -1, feat.shape[-2], feat.shape[-1])
            if self.mask_guidance == "attn_gate":
                feat = self.temporal_attn(feat, early_mask)
            else:
                feat = self.temporal_attn(feat)
            feat = feat.view(B * T, -1, feat.shape[-2], feat.shape[-1])

        # Skip gating: predict early mask and gate skip connections
        if self.mask_guidance == "skip_gate":
            early_mask = self.early_mask_conv(feat)  # [B*T, 1, H', W'] at bottleneck res
            gate = self.skip_gate_strength.clamp(0, 1)

            # Decoder with gated skip connections
            for decoder, skip in zip(self.decoders, reversed(skips)):
                # Upsample mask to match skip resolution
                skip_mask = F.interpolate(early_mask, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                # Gate: suppress encoder features in artifact regions
                # skip_gated = skip * (1 - gate * skip_mask)
                skip_gated = skip * (1 - gate * skip_mask)
                feat = decoder(feat, skip_gated)
        else:
            # Standard decoder with skip connections
            for decoder, skip in zip(self.decoders, reversed(skips)):
                feat = decoder(feat, skip)

        # Optional mask prediction (before output, so it can guide inpainting)
        mask = None
        if self.predict_mask:
            mask = self.mask_conv(feat)

            # Apply mask guidance if enabled
            if self.mask_guidance == "modulation":
                feat = self.mask_modulation(feat, mask)
            elif self.mask_guidance == "concat":
                feat = torch.cat([feat, mask], dim=1)

        # Output
        out = self.out_conv(feat)

        # Apply residual weighting if enabled (residual only in mask regions)
        if self.mask_guidance == "residual" and mask is not None:
            # out is the residual, apply only where mask > 0
            out = x_input + out * mask
            out = out.clamp(0, 1)
        elif self.residual_learning:
            out = x_input + out
            out = out.clamp(0, 1)

        # Reshape for video output
        if is_video:
            out = out.view(B, T, self.out_channels, H, W)

        # Return mask if predicted
        if self.predict_mask:
            if is_video:
                mask = mask.view(B, T, 1, H, W)
            return out, mask

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExplicitCompositeNet(nn.Module):
    """
    Explicit composite architecture for artifact removal.

    Predicts inpainted image and mask separately, then composites:
        output = input * (1 - mask) + inpainted * mask

    This explicitly preserves clean regions (mask=0) unchanged.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_channels: List[int] = [32, 64, 128, 256],
        use_temporal_attention: bool = True,
        num_frames: int = 5,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build base network (reuse TemporalNAFUNet components)
        self.in_conv = nn.Conv2d(in_channels, encoder_channels[0], 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                DownBlock(encoder_channels[i], encoder_channels[i + 1], num_blocks=2)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            NAFBlock(encoder_channels[-1]),
            NAFBlock(encoder_channels[-1]),
        )

        # Temporal attention
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(
                encoder_channels[-1], num_heads=attention_heads, num_frames=num_frames
            )

        # Decoder
        decoder_channels = list(reversed(encoder_channels))
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                UpBlock(decoder_channels[i], decoder_channels[i + 1], num_blocks=2)
            )

        # Separate heads for inpainting and mask
        self.inpaint_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], out_channels, 1),
            nn.Sigmoid(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0] // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C, H, W] video frames or [B, C, H, W] single frame

        Returns:
            Tuple of (output, mask)
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
        feat = self.in_conv(x_input)

        skips = []
        for encoder in self.encoders:
            feat, skip = encoder(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Temporal attention
        if self.use_temporal_attention and is_video:
            feat = feat.view(B, T, -1, feat.shape[-2], feat.shape[-1])
            feat = self.temporal_attn(feat)
            feat = feat.view(B * T, -1, feat.shape[-2], feat.shape[-1])

        # Decode
        for decoder, skip in zip(self.decoders, reversed(skips)):
            feat = decoder(feat, skip)

        # Predict inpainted and mask separately
        inpainted = self.inpaint_head(feat)
        mask = self.mask_head(feat)

        # Explicit composite: preserve clean regions exactly
        out = x_input * (1 - mask) + inpainted * mask

        # Reshape for video output
        if is_video:
            out = out.view(B, T, self.out_channels, H, W)
            mask = mask.view(B, T, 1, H, W)

        return out, mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class NAFUNetLite(nn.Module):
    """
    Lightweight U-Net with NAFNet-style blocks (no temporal attention).
    ~2-3M parameters, suitable for real-time on M4.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        residual_learning: bool = False,
    ):
        super().__init__()

        self.residual_learning = residual_learning

        channels = [16, 32, 64, 128]

        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Simple encoder
        self.enc1 = nn.Sequential(NAFBlock(channels[0]), nn.Conv2d(channels[0], channels[1], 2, stride=2))
        self.enc2 = nn.Sequential(NAFBlock(channels[1]), nn.Conv2d(channels[1], channels[2], 2, stride=2))
        self.enc3 = nn.Sequential(NAFBlock(channels[2]), nn.Conv2d(channels[2], channels[3], 2, stride=2))

        # Bottleneck
        self.bottleneck = NAFBlock(channels[3])

        # Simple decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2), NAFBlock(channels[2]))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2), NAFBlock(channels[1]))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2), NAFBlock(channels[0]))

        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x

        x = self.in_conv(x)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b) + e2
        d2 = self.dec2(d3) + e1
        d1 = self.dec1(d2) + x

        out = self.out_conv(d1)

        if self.residual_learning:
            out = x_input + out
            out = out.clamp(0, 1)

        return out




# Backward compatibility aliases
ArtifactRemovalNet = TemporalNAFUNet
ArtifactRemovalNetLite = NAFUNetLite


# Test
if __name__ == "__main__":
    # Test standard model
    model = TemporalNAFUNet(
        encoder_channels=[32, 64, 128, 256],
        use_temporal_attention=True,
        num_frames=5
    )
    print(f"Standard model parameters: {model.count_parameters():,}")

    # Test with video input
    x = torch.randn(2, 5, 3, 256, 256)
    y = model(x)
    print(f"Video input: {x.shape} -> output: {y.shape}")

    # Test with single frame
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Single frame: {x.shape} -> output: {y.shape}")

    # Test residual learning
    model_residual = TemporalNAFUNet(
        encoder_channels=[32, 64, 128, 256],
        residual_learning=True
    )
    x = torch.randn(2, 3, 256, 256).clamp(0, 1)
    y = model_residual(x)
    print(f"\nResidual learning: {x.shape} -> {y.shape}")

    # Test with mask prediction (multi-task)
    model_with_mask = TemporalNAFUNet(
        encoder_channels=[32, 64, 128, 256],
        predict_mask=True
    )
    x = torch.randn(2, 5, 3, 256, 256)
    y, mask = model_with_mask(x)
    print(f"With mask (multi-task): output {y.shape}, mask {mask.shape}")

    # Test lite model
    model_lite = NAFUNetLite()
    params = sum(p.numel() for p in model_lite.parameters())
    print(f"\nLite model parameters: {params:,}")

    x = torch.randn(1, 3, 256, 256)
    y = model_lite(x)
    print(f"Lite: {x.shape} -> {y.shape}")
