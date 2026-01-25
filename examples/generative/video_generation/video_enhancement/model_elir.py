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
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add time embedding
        h = h + self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.conv_skip(x)


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
# MMSE Network (Initial Latent Estimator)
# =============================================================================

class MMSEBlock(nn.Module):
    """Simple residual block for MMSE estimator."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h


class MMSENet(nn.Module):
    """
    Simple MMSE estimator for initial latent prediction.

    Predicts an initial estimate of clean latent from degraded latent.
    This provides a good starting point for the flow matching loop.
    """

    def __init__(self, in_channels: int = 16, hidden_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(*[MMSEBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        h = self.blocks(h)
        return x + self.out_conv(h)


# =============================================================================
# Mask Head (for composite output)
# =============================================================================

class MaskHead(nn.Module):
    """Predict artifact mask from input image."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.k_steps = k_steps
        self.t_emb_dim = t_emb_dim
        self.sigma_s = sigma_s
        self.dt = 1.0 / k_steps

        # Encoder: image -> latent
        self.encoder = TAESDEncoder(in_channels, latent_channels, hidden_channels)

        # MMSE: initial latent estimate (used at inference)
        self.mmse = MMSENet(latent_channels, hidden_channels, num_blocks=4)

        # Flow model: velocity prediction
        self.flow = LUnet(
            in_channels=latent_channels,
            hid_channels=flow_hidden_channels,
            out_channels=latent_channels,
            ch_mult=[1, 2, 1, 2],
            n_mid_blocks=3,
            t_emb_dim=t_emb_dim,
        )

        # Decoder: latent -> image
        self.decoder = TAESDDecoder(out_channels, latent_channels, hidden_channels)

        # Mask head: predict artifact regions from degraded input
        self.mask_head = MaskHead(in_channels, hidden_channels // 2)

        # Placeholder for temporal attention
        self.use_temporal_attention = use_temporal_attention

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
    ) -> torch.Tensor:
        """
        Compute flow matching loss for training.

        Args:
            degraded: Degraded images [B, C, H, W] or [B*T, C, H, W]
            clean: Clean images, same shape

        Returns:
            Flow matching loss (MSE between predicted and target velocity)
        """
        # Encode both to latent space
        with torch.no_grad():
            z_degraded = self.encode(degraded)
        z_clean = self.encode(clean)

        # Sample random timestep t in [0, 1] for each sample
        B = degraded.shape[0]
        t = torch.rand(B, device=degraded.device)

        # Interpolate: z_t = (1-t)*z_degraded + t*z_clean
        # Reshape t for broadcasting: [B, 1, 1, 1]
        t_broadcast = t.view(B, 1, 1, 1)
        z_t = (1 - t_broadcast) * z_degraded + t_broadcast * z_clean

        # Predict velocity
        v_pred = self.predict_velocity(z_t, t)

        # Target velocity: straight path from degraded to clean
        v_target = z_clean - z_degraded

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

        # Predict mask from degraded input
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
        out = x_input * (1 - composite_mask) + restored * composite_mask

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

        Args:
            path: Path to ELIR checkpoint
            strict: If False, ignore missing/unexpected keys (mask_head, etc.)
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
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
                print(f"  Missing keys (will be randomly initialized): {missing[:5]}...")
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
