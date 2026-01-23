"""
Loss Functions for Video Artifact Removal

Includes pixel, perceptual, temporal, and frequency losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (differentiable L1)"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.
    Lighter alternative to LPIPS when that's not available.
    """
    def __init__(self, layers: Tuple[int, ...] = (3, 8, 15, 22)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.layers = layers
        self.features = nn.ModuleList()
        
        prev = 0
        for layer in layers:
            self.features.append(vgg[prev:layer + 1])
            prev = layer + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x_pred, x_target = pred, target
        
        for feature in self.features:
            x_pred = feature(x_pred)
            x_target = feature(x_target)
            loss += F.l1_loss(x_pred, x_target)
        
        return loss / len(self.features)


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video.
    Penalizes frame-to-frame differences that don't exist in ground truth.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_frames: [B, T, C, H, W]
            target_frames: [B, T, C, H, W]
        """
        # Compute temporal differences
        pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
        target_diff = target_frames[:, 1:] - target_frames[:, :-1]
        
        return F.l1_loss(pred_diff, target_diff)


class FFTLoss(nn.Module):
    """
    Frequency domain loss.
    Helps recover fine textures and high-frequency details.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Compare magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return F.l1_loss(pred_mag, target_mag)


class EdgeLoss(nn.Module):
    """
    Edge preservation loss using Sobel filters.
    """
    def __init__(self):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute edges
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_edge = torch.abs(pred_edge_x) + torch.abs(pred_edge_y)
        
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_edge = torch.abs(target_edge_x) + torch.abs(target_edge_y)
        
        return F.l1_loss(pred_edge, target_edge)


class ArtifactRemovalLoss(nn.Module):
    """
    Combined loss for artifact removal.
    """
    def __init__(
        self,
        w_pixel: float = 1.0,
        w_perceptual: float = 0.1,
        w_temporal: float = 0.5,
        w_fft: float = 0.05,
        w_edge: float = 0.0,
        use_lpips: bool = True,
    ):
        super().__init__()
        
        self.w_pixel = w_pixel
        self.w_perceptual = w_perceptual
        self.w_temporal = w_temporal
        self.w_fft = w_fft
        self.w_edge = w_edge
        
        # Losses
        self.pixel_loss = CharbonnierLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.fft_loss = FFTLoss()
        self.edge_loss = EdgeLoss() if w_edge > 0 else None
        
        # Perceptual loss
        self.use_lpips = use_lpips
        if use_lpips:
            try:
                import lpips
                self.perceptual_loss = lpips.LPIPS(net='alex')
                # Freeze LPIPS
                for param in self.perceptual_loss.parameters():
                    param.requires_grad = False
            except ImportError:
                print("LPIPS not available, falling back to VGG perceptual loss")
                self.use_lpips = False
                self.perceptual_loss = VGGPerceptualLoss()
        else:
            self.perceptual_loss = VGGPerceptualLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        is_video: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            pred: [B, T, C, H, W] or [B, C, H, W]
            target: same shape as pred
            is_video: whether input is video (for temporal loss)
        
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Handle video vs image input
        if is_video and pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred_flat = pred.view(B * T, C, H, W)
            target_flat = target.view(B * T, C, H, W)
        else:
            pred_flat = pred
            target_flat = target
        
        # Pixel loss
        losses['pixel'] = self.w_pixel * self.pixel_loss(pred_flat, target_flat)
        
        # Perceptual loss
        if self.use_lpips:
            # LPIPS expects [-1, 1] range
            pred_norm = pred_flat * 2 - 1
            target_norm = target_flat * 2 - 1
            losses['perceptual'] = self.w_perceptual * self.perceptual_loss(
                pred_norm, target_norm
            ).mean()
        else:
            losses['perceptual'] = self.w_perceptual * self.perceptual_loss(
                pred_flat, target_flat
            )
        
        # FFT loss
        if self.w_fft > 0:
            losses['fft'] = self.w_fft * self.fft_loss(pred_flat, target_flat)
        
        # Edge loss
        if self.edge_loss is not None and self.w_edge > 0:
            losses['edge'] = self.w_edge * self.edge_loss(pred_flat, target_flat)
        
        # Temporal consistency loss (only for video)
        if is_video and pred.dim() == 5 and self.w_temporal > 0:
            losses['temporal'] = self.w_temporal * self.temporal_loss(pred, target)
        
        # Total loss
        losses['total'] = sum(v for k, v in losses.items() if k != 'total')
        
        return losses




class PSNRMetric:
    """PSNR metric for evaluation"""
    @staticmethod
    def compute(pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()


class SSIMMetric:
    """Simplified SSIM metric"""
    @staticmethod
    def compute(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> float:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Simple Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        window = g.outer(g)
        window = window / window.sum()
        window = window.view(1, 1, window_size, window_size).to(pred.device)
        window = window.expand(pred.shape[1], 1, window_size, window_size)
        
        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()


# Test
if __name__ == "__main__":
    # Test losses
    loss_fn = ArtifactRemovalLoss(use_lpips=False)  # Use VGG for testing
    
    # Single frame
    pred = torch.randn(2, 3, 256, 256).clamp(0, 1)
    target = torch.randn(2, 3, 256, 256).clamp(0, 1)
    
    losses = loss_fn(pred, target, is_video=False)
    print("Single frame losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Video
    pred = torch.randn(2, 5, 3, 256, 256).clamp(0, 1)
    target = torch.randn(2, 5, 3, 256, 256).clamp(0, 1)
    
    losses = loss_fn(pred, target, is_video=True)
    print("\nVideo losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Metrics
    print(f"\nPSNR: {PSNRMetric.compute(pred, target):.2f} dB")
    print(f"SSIM: {SSIMMetric.compute(pred.view(-1, 3, 256, 256), target.view(-1, 3, 256, 256)):.4f}")
