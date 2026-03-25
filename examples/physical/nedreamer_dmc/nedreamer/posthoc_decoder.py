"""
Post-hoc Decoder/Renderer for Decoder-free Dreamer Variants

This module provides a visualization decoder that is trained AFTER the agent,
purely for interpretation purposes. Gradients do NOT flow to the world model
or policy - this is a separate model for rendering latent states to images.

Use cases:
1. Posterior rendering: Visualize what the agent believes it's seeing
2. Imagination rendering: Visualize predicted futures under action sequences  
3. Counterfactual visualization: Compare imagined outcomes of different actions
4. Value-guided generation: Show what high-value states look like

Based on:
- "Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction"
- Post-hoc visualization techniques for decoder-free world models
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class ConvDecoder(nn.Module):
    """Convolutional decoder for rendering latent states to images.
    
    Maps (stoch, deter) latent state to image observation.
    Uses transposed convolutions with residual connections.
    """
    
    def __init__(
        self,
        feat_size: int,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        hidden_dim: int = 256,
        depth: int = 32,
        min_res: int = 4,
    ):
        """
        Args:
            feat_size: Size of flattened latent feature (stoch * discrete + deter)
            image_shape: Output image shape (H, W, C)
            hidden_dim: Hidden layer dimension
            depth: Base channel depth for conv layers
            min_res: Minimum spatial resolution before upsampling
        """
        super().__init__()
        
        self.image_shape = image_shape
        H, W, C = image_shape
        self.min_res = min_res
        
        # Calculate number of upsampling steps needed
        self.num_ups = 0
        res = min_res
        while res < H:
            self.num_ups += 1
            res *= 2
        
        # Channel depths for each resolution level (highest to lowest res)
        self.depths = [depth * (2 ** i) for i in range(self.num_ups, -1, -1)]
        
        # Initial projection from latent to spatial feature map
        self.init_channels = self.depths[0]
        self.fc = nn.Sequential(
            nn.Linear(feat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.init_channels * min_res * min_res),
            nn.LayerNorm(self.init_channels * min_res * min_res),
            nn.SiLU(),
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = self.init_channels
        for i, out_ch in enumerate(self.depths[1:]):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch
        
        # Final output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch, C, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: Latent features (B, feat_size) or (B, T, feat_size)
            
        Returns:
            Reconstructed images (B, H, W, C) or (B, T, H, W, C) in [0, 1]
        """
        # Handle sequence dimension
        has_time = feat.dim() == 3
        if has_time:
            B, T, D = feat.shape
            feat = feat.reshape(B * T, D)
        
        # Project to spatial features
        x = self.fc(feat)  # (B, init_channels * min_res * min_res)
        x = x.view(-1, self.init_channels, self.min_res, self.min_res)
        
        # Upsample
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # Output
        x = self.out_conv(x)  # (B, C, H, W)
        
        # Convert to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # Restore time dimension if needed
        if has_time:
            x = x.reshape(B, T, *x.shape[1:])
        
        return x


class PostHocDecoder(nn.Module):
    """Post-hoc decoder manager for decoder-free Dreamer variants.
    
    Handles training, saving, loading, and visualization generation.
    Gradients are completely isolated from the world model.
    """
    
    def __init__(
        self,
        feat_size: int,
        image_shape: Tuple[int, int, int] = (64, 64, 3),
        hidden_dim: int = 256,
        depth: int = 32,
        lr: float = 1e-4,
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.feat_size = feat_size
        self.image_shape = image_shape
        self.device = torch.device(device)
        
        # Create decoder
        self.decoder = ConvDecoder(
            feat_size=feat_size,
            image_shape=image_shape,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        
        # Training stats
        self.train_steps = 0
        self.loss_ema = 0.0
    
    @torch.no_grad()
    def get_features(
        self,
        world_model,
        stoch: torch.Tensor,
        deter: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from latent states (no gradients to world model).
        
        Args:
            world_model: Dreamer world model (for get_feat method)
            stoch: Stochastic state (B, stoch_dim, discrete_dim) or (B, T, ...)
            deter: Deterministic state (B, deter_dim) or (B, T, deter_dim)
            
        Returns:
            Features tensor (B, feat_size) or (B, T, feat_size)
        """
        return world_model.dynamics.get_feat(stoch, deter)
    
    def train_step(
        self,
        images: torch.Tensor,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        world_model,
    ) -> Dict[str, float]:
        """Single training step for the post-hoc decoder.
        
        Args:
            images: Target images (B, H, W, C) or (B, T, H, W, C) in [0, 1]
            stoch: Stochastic states (detached from world model)
            deter: Deterministic states (detached from world model)
            world_model: World model for feature extraction
            
        Returns:
            Dict of training metrics
        """
        self.decoder.train()
        
        # Get features (no grad to world model)
        with torch.no_grad():
            feat = self.get_features(world_model, stoch.detach(), deter.detach())
        
        # Forward pass through decoder
        recon = self.decoder(feat)
        
        # Normalize target images to [0, 1] if needed
        if images.max() > 1.0:
            images = images.float() / 255.0
        
        # Reconstruction loss (MSE)
        loss = F.mse_loss(recon, images)
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Update stats
        self.train_steps += 1
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss.item()
        
        return {
            'posthoc_decoder/loss': loss.item(),
            'posthoc_decoder/loss_ema': self.loss_ema,
            'posthoc_decoder/train_steps': self.train_steps,
        }
    
    @torch.no_grad()
    def render_posterior(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        world_model,
    ) -> torch.Tensor:
        """Render images from posterior latent states.
        
        Args:
            stoch: Posterior stochastic states
            deter: Posterior deterministic states
            world_model: World model for feature extraction
            
        Returns:
            Rendered images (B, H, W, C) or (B, T, H, W, C) in [0, 1]
        """
        self.decoder.eval()
        feat = self.get_features(world_model, stoch, deter)
        return self.decoder(feat)
    
    @torch.no_grad()
    def render_imagination(
        self,
        init_stoch: torch.Tensor,
        init_deter: torch.Tensor,
        actions: torch.Tensor,
        world_model,
    ) -> torch.Tensor:
        """Render imagined future trajectory.
        
        Args:
            init_stoch: Initial stochastic state (B, stoch_dim, discrete_dim)
            init_deter: Initial deterministic state (B, deter_dim)
            actions: Action sequence (B, T, act_dim)
            world_model: World model for imagination
            
        Returns:
            Rendered imagined images (B, T, H, W, C) in [0, 1]
        """
        self.decoder.eval()
        
        # Imagine forward using world model prior dynamics
        stoch, deter = world_model.dynamics.imagine_with_action(
            init_stoch, init_deter, actions
        )
        
        # Render imagined states
        feat = self.get_features(world_model, stoch, deter)
        return self.decoder(feat)
    
    @torch.no_grad()
    def render_counterfactual(
        self,
        init_stoch: torch.Tensor,
        init_deter: torch.Tensor,
        action_sequences: List[torch.Tensor],
        world_model,
    ) -> List[torch.Tensor]:
        """Render multiple counterfactual futures from same initial state.
        
        Args:
            init_stoch: Initial stochastic state (B, ...)
            init_deter: Initial deterministic state (B, ...)
            action_sequences: List of action sequences [(B, T, act_dim), ...]
            world_model: World model for imagination
            
        Returns:
            List of rendered trajectories [(B, T, H, W, C), ...]
        """
        results = []
        for actions in action_sequences:
            rendered = self.render_imagination(
                init_stoch, init_deter, actions, world_model
            )
            results.append(rendered)
        return results
    
    @torch.no_grad()
    def render_open_loop_prediction(
        self,
        context_images: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        world_model,
        preprocess_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render open-loop prediction: context frames → imagined future.
        
        This is the key visualization for decoder-free Dreamers:
        1. Process K context frames to infer latent state (filtering)
        2. Roll out H steps using only actions (open-loop imagination)
        3. Render the imagined trajectory
        
        Args:
            context_images: Context observations (K, H, W, C) in [0, 255]
            context_actions: Actions during context (K, act_dim)
            future_actions: Actions for future prediction (H, act_dim)
            world_model: World model for encoding and imagination
            preprocess_fn: Preprocessing function for observations
            
        Returns:
            context_renders: Rendered context frames (K, H, W, C) in [0, 1]
            future_renders: Rendered imagined future (H, H, W, C) in [0, 1]
            context_states: (stoch_seq, deter_seq) for the context
        """
        self.decoder.eval()
        K = context_images.shape[0]
        H_pred = future_actions.shape[0]
        device = context_images.device
        
        # === 1. Process context frames (filtering) ===
        stoch, deter = world_model.dynamics.initial(1)
        prev_action = torch.zeros(1, context_actions.shape[-1], device=device)
        
        context_stoch_list = []
        context_deter_list = []
        
        for t in range(K):
            # Prepare observation
            obs = {'image': context_images[t:t+1]}  # (1, H, W, C)
            is_first = torch.tensor([t == 0], device=device)
            
            if t == 0:
                stoch, deter = world_model.dynamics.initial(1)
                prev_action = torch.zeros(1, context_actions.shape[-1], device=device)
            
            # Encode and update state (posterior update)
            p_obs = preprocess_fn(obs)
            embed = world_model.encoder(p_obs)
            stoch, deter, _ = world_model.dynamics.obs_step(
                stoch, deter, prev_action, embed, is_first
            )
            
            context_stoch_list.append(stoch.clone())
            context_deter_list.append(deter.clone())
            prev_action = context_actions[t:t+1]
        
        # Stack context states
        context_stoch = torch.cat(context_stoch_list, dim=0)  # (K, stoch, discrete)
        context_deter = torch.cat(context_deter_list, dim=0)  # (K, deter)
        
        # Render context frames
        context_feat = self.get_features(world_model, context_stoch, context_deter)
        context_renders = self.decoder(context_feat)  # (K, H, W, C)
        
        # === 2. Open-loop imagination (no observations, just actions) ===
        # Start from the last context state
        init_stoch = context_stoch_list[-1]  # (1, stoch, discrete)
        init_deter = context_deter_list[-1]  # (1, deter)
        
        # Imagine forward using only actions
        future_stoch, future_deter = world_model.dynamics.imagine_with_action(
            init_stoch, init_deter, future_actions.unsqueeze(0)
        )  # (1, H, stoch, discrete), (1, H, deter)
        
        # Render imagined future
        future_feat = self.get_features(
            world_model, future_stoch.squeeze(0), future_deter.squeeze(0)
        )
        future_renders = self.decoder(future_feat)  # (H, H, W, C)
        
        return context_renders, future_renders, (context_stoch, context_deter)
    
    @torch.no_grad()
    def render_open_loop_with_uncertainty(
        self,
        context_images: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        world_model,
        preprocess_fn,
        num_samples: int = 4,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Render open-loop prediction with multiple samples to show uncertainty.
        
        Runs multiple imagination rollouts from the same context to visualize
        the uncertainty in the world model's predictions.
        
        Args:
            context_images: Context observations (K, H, W, C)
            context_actions: Actions during context (K, act_dim)
            future_actions: Actions for future prediction (H, act_dim)
            world_model: World model
            preprocess_fn: Preprocessing function
            num_samples: Number of samples to generate
            
        Returns:
            context_renders: Single context rendering (K, H, W, C)
            future_samples: List of future renderings [(H, H, W, C), ...] * num_samples
        """
        self.decoder.eval()
        K = context_images.shape[0]
        device = context_images.device
        
        # Process context once
        stoch, deter = world_model.dynamics.initial(1)
        prev_action = torch.zeros(1, context_actions.shape[-1], device=device)
        
        context_stoch_list = []
        context_deter_list = []
        
        for t in range(K):
            obs = {'image': context_images[t:t+1]}
            is_first = torch.tensor([t == 0], device=device)
            
            if t == 0:
                stoch, deter = world_model.dynamics.initial(1)
                prev_action = torch.zeros(1, context_actions.shape[-1], device=device)
            
            p_obs = preprocess_fn(obs)
            embed = world_model.encoder(p_obs)
            stoch, deter, _ = world_model.dynamics.obs_step(
                stoch, deter, prev_action, embed, is_first
            )
            
            context_stoch_list.append(stoch.clone())
            context_deter_list.append(deter.clone())
            prev_action = context_actions[t:t+1]
        
        # Render context
        context_stoch = torch.cat(context_stoch_list, dim=0)
        context_deter = torch.cat(context_deter_list, dim=0)
        context_feat = self.get_features(world_model, context_stoch, context_deter)
        context_renders = self.decoder(context_feat)
        
        # Generate multiple future samples
        init_stoch = context_stoch_list[-1]
        init_deter = context_deter_list[-1]
        
        future_samples = []
        for _ in range(num_samples):
            # Each sample uses stochastic transitions in imagine_with_action
            future_stoch, future_deter = world_model.dynamics.imagine_with_action(
                init_stoch, init_deter, future_actions.unsqueeze(0)
            )
            
            future_feat = self.get_features(
                world_model, future_stoch.squeeze(0), future_deter.squeeze(0)
            )
            future_render = self.decoder(future_feat)
            future_samples.append(future_render)
        
        return context_renders, future_samples
    
    @torch.no_grad()
    def render_value_gradient(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        world_model,
        critic,
        num_steps: int = 50,
        lr: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate high-value state visualization via latent optimization.
        
        Optimizes the latent to increase value, then renders.
        Stays on-manifold by regularizing toward original latent.
        
        Args:
            stoch: Starting stochastic state
            deter: Starting deterministic state  
            world_model: World model
            critic: Value critic
            num_steps: Optimization steps
            lr: Learning rate for latent optimization
            
        Returns:
            original_render: Original state rendering
            optimized_render: High-value state rendering
        """
        self.decoder.eval()
        
        # Get initial features and value
        init_feat = self.get_features(world_model, stoch, deter)
        original_render = self.decoder(init_feat)
        
        # Make feat optimizable
        opt_feat = init_feat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt_feat], lr=lr)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            
            # Value from critic (treat feat as if it came from world model)
            value = critic.value(opt_feat).mode()
            
            # Maximize value, regularize toward original
            reg_loss = 0.1 * F.mse_loss(opt_feat, init_feat.detach())
            loss = -value.mean() + reg_loss
            
            loss.backward()
            optimizer.step()
        
        # Render optimized latent
        with torch.no_grad():
            optimized_render = self.decoder(opt_feat)
        
        return original_render, optimized_render
    
    def save(self, path: Path):
        """Save decoder checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'loss_ema': self.loss_ema,
            'feat_size': self.feat_size,
            'image_shape': self.image_shape,
        }, path)
        print(f"[PostHocDecoder] Saved checkpoint to {path}")
    
    def load(self, path: Path):
        """Load decoder checkpoint."""
        path = Path(path)
        if not path.exists():
            print(f"[PostHocDecoder] No checkpoint found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
        self.loss_ema = checkpoint['loss_ema']
        print(f"[PostHocDecoder] Loaded checkpoint from {path} (step {self.train_steps})")
        return True


def create_comparison_video(
    original_images: np.ndarray,
    posterior_renders: np.ndarray,
    imagined_renders: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create side-by-side comparison video.
    
    Args:
        original_images: Ground truth images (T, H, W, C)
        posterior_renders: Posterior reconstructions (T, H, W, C)
        imagined_renders: Optional imagined future (T, H, W, C)
        
    Returns:
        Combined video (T, H, combined_W, C)
    """
    # Ensure uint8
    def to_uint8(x):
        if x.max() <= 1.0:
            x = (x * 255).clip(0, 255)
        return x.astype(np.uint8)
    
    original = to_uint8(original_images)
    posterior = to_uint8(posterior_renders)
    
    frames = [original, posterior]
    
    if imagined_renders is not None:
        imagined = to_uint8(imagined_renders)
        frames.append(imagined)
    
    # Concatenate horizontally
    return np.concatenate(frames, axis=2)


def create_counterfactual_grid(
    init_image: np.ndarray,
    counterfactual_renders: List[np.ndarray],
    action_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Create grid comparing counterfactual futures.
    
    Args:
        init_image: Initial observation (H, W, C)
        counterfactual_renders: List of future trajectories [(T, H, W, C), ...]
        action_labels: Optional labels for each action sequence
        
    Returns:
        Grid image showing initial state + future trajectories
    """
    def to_uint8(x):
        if x.max() <= 1.0:
            x = (x * 255).clip(0, 255)
        return x.astype(np.uint8)
    
    init = to_uint8(init_image)
    H, W, C = init.shape
    
    num_futures = len(counterfactual_renders)
    T = counterfactual_renders[0].shape[0]
    
    # Create grid: rows = different action sequences, cols = timesteps
    # First column is initial state (repeated)
    grid_h = num_futures * H
    grid_w = (T + 1) * W  # +1 for initial state column
    
    grid = np.zeros((grid_h, grid_w, C), dtype=np.uint8)
    
    for i, renders in enumerate(counterfactual_renders):
        renders = to_uint8(renders)
        y_offset = i * H
        
        # Initial state
        grid[y_offset:y_offset+H, 0:W] = init
        
        # Future frames
        for t in range(T):
            x_offset = (t + 1) * W
            grid[y_offset:y_offset+H, x_offset:x_offset+W] = renders[t]
    
    return grid

