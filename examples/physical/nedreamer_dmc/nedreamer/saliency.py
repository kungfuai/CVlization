"""
Saliency Maps for Vision-based RL Agents

Implementation based on Greydanus et al., 2018 "Visualizing and Understanding Atari Agents"
Adapted for DreamerV3-style world model agents.

The core idea is perturbation-based saliency: locally blur small regions of the
observation and measure how much the agent's policy and value outputs change.
This reveals which parts of the observation are causally important for decisions.

Key differences from Atari setting:
- DreamerV3 acts on latent states from world model (RSSM), not directly on pixels
- Perturbation affects posterior inference (belief formation) + action selection
- This captures what pixels matter for BOTH state estimation AND policy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


def gaussian_blur_2d(obs: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """Apply Gaussian blur to image tensor.
    
    Args:
        obs: Image tensor of shape (..., H, W, C) with values in [0, 1]
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Blurred image of same shape
    """
    # Compute kernel size (6*sigma covers ~99.7% of distribution)
    kernel_size = int(6 * sigma) | 1  # Ensure odd
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=obs.device, dtype=obs.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel via outer product
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    # Store original shape and reshape for conv2d
    original_shape = obs.shape
    H, W, C = original_shape[-3:]
    
    # Reshape to (N, C, H, W) for conv2d
    obs_flat = obs.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (N, C, H, W)
    N = obs_flat.shape[0]
    
    # Apply blur per channel (groups=C for depthwise conv)
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    
    blurred = F.conv2d(
        obs_flat.reshape(N * C, 1, H, W),
        kernel_2d[:1],  # Use same kernel for all channels
        padding=padding
    ).reshape(N, C, H, W)
    
    # Reshape back to original format
    blurred = blurred.permute(0, 2, 3, 1)  # (N, H, W, C)
    blurred = blurred.reshape(original_shape)
    
    return blurred


def create_gaussian_mask(
    H: int, 
    W: int, 
    center_y: int, 
    center_x: int, 
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a 2D Gaussian mask centered at (center_y, center_x).
    
    Args:
        H, W: Image dimensions
        center_y, center_x: Center coordinates of the mask
        sigma: Standard deviation of Gaussian
        device: Torch device
        dtype: Tensor dtype
        
    Returns:
        Mask tensor of shape (H, W, 1) with values in [0, 1]
    """
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2
    mask = torch.exp(-dist_sq / (2 * sigma ** 2))
    
    return mask.unsqueeze(-1)  # (H, W, 1)


def localized_blur(
    obs: torch.Tensor,
    center_yx: Tuple[int, int],
    mask_sigma: float = 5.0,
    blur_sigma: float = 3.0,
    blurred_obs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Apply localized blur to observation at specified center.
    
    Implements the perturbation from Greydanus et al.:
    obs' = obs * (1 - M) + blur(obs) * M
    
    where M is a Gaussian mask centered at center_yx.
    
    Args:
        obs: Image tensor of shape (H, W, C) with values in [0, 1]
        center_yx: (y, x) center coordinates for the blur
        mask_sigma: Standard deviation of the Gaussian mask
        blur_sigma: Standard deviation of the Gaussian blur
        blurred_obs: Pre-computed blurred observation (optional, for efficiency)
        
    Returns:
        Perturbed observation of same shape
    """
    H, W, C = obs.shape
    center_y, center_x = center_yx
    
    # Create Gaussian mask
    mask = create_gaussian_mask(H, W, center_y, center_x, mask_sigma, obs.device, obs.dtype)
    
    # Compute or use pre-computed blur
    if blurred_obs is None:
        blurred_obs = gaussian_blur_2d(obs, blur_sigma)
    
    # Interpolate between original and blurred
    perturbed = obs * (1.0 - mask) + blurred_obs * mask
    
    return perturbed


def create_perturbation_batch(
    obs: torch.Tensor,
    stride: int = 5,
    mask_sigma: float = 5.0,
    blur_sigma: float = 3.0
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Create batch of perturbed observations for saliency computation.
    
    Args:
        obs: Single observation of shape (H, W, C)
        stride: Grid stride for perturbation centers
        mask_sigma: Gaussian mask sigma
        blur_sigma: Gaussian blur sigma
        
    Returns:
        perturbed_batch: Tensor of shape (num_perturbations, H, W, C)
        grid_shape: (Gh, Gw) grid dimensions
    """
    H, W, C = obs.shape
    device = obs.device
    dtype = obs.dtype
    
    # Pre-compute blurred observation for efficiency
    blurred_obs = gaussian_blur_2d(obs, blur_sigma)
    
    # Generate grid of centers
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    Gh, Gw = len(ys), len(xs)
    
    # Create batch of perturbed observations
    perturbed_list = []
    for y in ys:
        for x in xs:
            perturbed = localized_blur(obs, (y, x), mask_sigma, blur_sigma, blurred_obs)
            perturbed_list.append(perturbed)
    
    perturbed_batch = torch.stack(perturbed_list, dim=0)  # (B, H, W, C)
    
    return perturbed_batch, (Gh, Gw)


@torch.no_grad()
def compute_saliency_step(
    world_model,
    actor_critic,
    obs: Dict,
    prev_stoch: torch.Tensor,
    prev_deter: torch.Tensor,
    prev_action: torch.Tensor,
    is_first: torch.Tensor,
    preprocess_fn,
    stride: int = 5,
    mask_sigma: float = 5.0,
    blur_sigma: float = 3.0,
    act_discrete: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute actor and critic saliency maps for a single observation.
    
    This function:
    1. Computes baseline posterior state and policy/value outputs
    2. Creates a batch of locally perturbed observations
    3. Recomputes posterior and outputs for each perturbation
    4. Measures L2 divergence in actor parameters and value
    
    Args:
        world_model: DreamerV3 world model (encoder + RSSM + heads)
        actor_critic: Actor-critic module
        obs: Raw observation dict with 'image' key. Image can have shape 
             (1, H, W, C) or (H, W, C), values in 0-255
        prev_stoch: Previous stochastic state (stoch_dim, discrete_dim)
        prev_deter: Previous deterministic state (deter_dim,)
        prev_action: Previous action (act_dim,)
        is_first: Boolean tensor indicating episode start
        preprocess_fn: Function to preprocess observation dict
        stride: Grid stride for saliency computation
        mask_sigma: Gaussian mask sigma
        blur_sigma: Gaussian blur sigma
        act_discrete: Whether action space is discrete
        
    Returns:
        actor_saliency: (Gh, Gw) tensor of actor saliency scores
        critic_saliency: (Gh, Gw) tensor of critic saliency scores
    """
    device = prev_stoch.device
    
    # Get image from observation (handle dict format)
    image = obs['image']  # (1, H, W, C) or (H, W, C) with values 0-255
    original_image = image.clone()
    
    # Remove batch dimension if present for perturbation processing
    if image.dim() == 4:
        image = image.squeeze(0)  # (H, W, C)
    
    # Normalize to [0, 1] for perturbation
    if image.max() > 1.0:
        image = image.float() / 255.0
    else:
        image = image.float()
    
    H, W, C = image.shape
    
    # === 1. Baseline computation ===
    # Ensure obs has batch dimension for encoder
    obs_batched = {}
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            if val.dim() == 3 and key == 'image':  # (H, W, C)
                obs_batched[key] = val.unsqueeze(0)
            elif val.dim() == 0:  # scalar
                obs_batched[key] = val.unsqueeze(0)
            elif val.dim() == 1:  # (D,)
                obs_batched[key] = val.unsqueeze(0)
            else:
                obs_batched[key] = val
        else:
            obs_batched[key] = val
    
    p_obs = preprocess_fn(obs_batched)
    embed = world_model.encoder(p_obs)  # (1, embed_dim)
    
    # Ensure states have proper batch dimension
    prev_stoch_b = prev_stoch.unsqueeze(0) if prev_stoch.dim() == 2 else prev_stoch
    prev_deter_b = prev_deter.unsqueeze(0) if prev_deter.dim() == 1 else prev_deter
    prev_action_b = prev_action.unsqueeze(0) if prev_action.dim() == 1 else prev_action
    # Handle is_first which may have various shapes: scalar, (1,), (1, 1), etc.
    # Flatten to scalar then add batch dim to get (1,)
    is_first_scalar = is_first.flatten()[0] if is_first.numel() > 0 else is_first
    is_first_b = is_first_scalar.unsqueeze(0)
    
    # Compute posterior state
    stoch, deter, _ = world_model.dynamics.obs_step(
        prev_stoch_b, prev_deter_b, prev_action_b, embed, is_first_b
    )
    
    # Get features for actor/critic
    feat = world_model.dynamics.get_feat(stoch, deter)  # (1, feat_dim)
    
    # Baseline actor outputs (distribution parameters)
    base_actor_dist = actor_critic.actor(feat)
    if act_discrete:
        # Discrete: use logits
        base_actor_params = base_actor_dist.logits  # (1, act_dim)
    else:
        # Continuous: use mean (pre-tanh) from Normal distribution
        # bounded_normal returns Independent(Normal(...), 1)
        base_actor_params = base_actor_dist.base_dist.mean  # (1, act_dim)
    
    # Baseline value
    base_value = actor_critic.value(feat).mode()  # (1, 1)
    
    # === 2. Create perturbation batch ===
    perturbed_batch, (Gh, Gw) = create_perturbation_batch(
        image, stride, mask_sigma, blur_sigma
    )
    num_perturbations = Gh * Gw
    
    # === 3. Batch recomputation ===
    # Reconstruct observation dict for perturbed batch
    # Scale back to 0-255 if original was in that range
    orig_img = obs['image']
    orig_max = orig_img.max() if isinstance(orig_img, torch.Tensor) else 1.0
    if orig_max > 1.0:
        perturbed_images = (perturbed_batch * 255.0).to(orig_img.dtype)
    else:
        perturbed_images = perturbed_batch
    
    # Create batched observation dict
    perturbed_obs = {}
    for key, val in obs.items():
        if key == 'image':
            perturbed_obs[key] = perturbed_images  # (B, H, W, C)
        else:
            # Broadcast other keys to batch size
            if isinstance(val, torch.Tensor):
                # Handle various dimensions
                if val.dim() == 0:  # scalar
                    perturbed_obs[key] = val.unsqueeze(0).expand(num_perturbations)
                elif val.dim() == 1:  # (D,) or (1,)
                    if val.shape[0] == 1:  # batched scalar
                        perturbed_obs[key] = val.expand(num_perturbations, 1).squeeze(-1)
                    else:  # feature vector
                        perturbed_obs[key] = val.unsqueeze(0).expand(num_perturbations, -1)
                elif val.dim() == 2:  # (1, D)
                    perturbed_obs[key] = val.expand(num_perturbations, -1)
                elif val.dim() == 4 and key != 'image':  # (1, H, W, C)
                    perturbed_obs[key] = val.expand(num_perturbations, -1, -1, -1)
                else:
                    perturbed_obs[key] = val.unsqueeze(0).expand(num_perturbations, *val.shape)
            else:
                perturbed_obs[key] = val
    
    # Preprocess batch
    p_obs_batch = preprocess_fn(perturbed_obs)
    
    # Encode batch
    embed_batch = world_model.encoder(p_obs_batch)  # (B, embed_dim)
    
    # Broadcast previous state to batch
    prev_stoch_batch = prev_stoch.unsqueeze(0).expand(num_perturbations, *prev_stoch.shape)
    prev_deter_batch = prev_deter.unsqueeze(0).expand(num_perturbations, *prev_deter.shape)
    prev_action_batch = prev_action.unsqueeze(0).expand(num_perturbations, *prev_action.shape)
    # Handle is_first which may have various shapes: scalar, (1,), (1, 1), etc.
    is_first_scalar = is_first.flatten()[0] if is_first.numel() > 0 else is_first
    is_first_batch = is_first_scalar.unsqueeze(0).expand(num_perturbations)
    
    # Compute posterior for perturbed batch
    stoch_batch, deter_batch, _ = world_model.dynamics.obs_step(
        prev_stoch_batch, prev_deter_batch, prev_action_batch,
        embed_batch, is_first_batch
    )
    
    # Get features for batch
    feat_batch = world_model.dynamics.get_feat(stoch_batch, deter_batch)  # (B, feat_dim)
    
    # Actor outputs for batch
    actor_dist_batch = actor_critic.actor(feat_batch)
    if act_discrete:
        actor_params_batch = actor_dist_batch.logits
    else:
        actor_params_batch = actor_dist_batch.base_dist.mean
    
    # Value for batch
    value_batch = actor_critic.value(feat_batch).mode()  # (B, 1)
    
    # === 4. Compute saliency scores ===
    # Actor saliency: L2 divergence of action distribution parameters
    actor_diff = actor_params_batch - base_actor_params.expand_as(actor_params_batch)
    actor_saliency = 0.5 * (actor_diff ** 2).sum(dim=-1)  # (B,)
    
    # Critic saliency: squared difference in value
    value_diff = value_batch.squeeze(-1) - base_value.squeeze(-1).expand(num_perturbations)
    critic_saliency = 0.5 * (value_diff ** 2)  # (B,)
    
    # Reshape to grid
    actor_saliency = actor_saliency.view(Gh, Gw)
    critic_saliency = critic_saliency.view(Gh, Gw)
    
    return actor_saliency, critic_saliency


def upsample_saliency(saliency: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Upsample saliency map to target image size using bilinear interpolation.
    
    Args:
        saliency: Saliency grid of shape (Gh, Gw)
        target_size: (H, W) target dimensions
        
    Returns:
        Upsampled saliency of shape (H, W)
    """
    H, W = target_size
    # Add batch and channel dims for F.interpolate
    saliency = saliency.unsqueeze(0).unsqueeze(0)  # (1, 1, Gh, Gw)
    upsampled = F.interpolate(saliency, size=(H, W), mode='bilinear', align_corners=False)
    return upsampled.squeeze(0).squeeze(0)  # (H, W)


def normalize_saliency(saliency: torch.Tensor, percentile: float = 99.0) -> torch.Tensor:
    """Normalize saliency map to [0, 1] range with percentile clipping.
    
    Args:
        saliency: Saliency tensor of any shape
        percentile: Percentile for clipping high values
        
    Returns:
        Normalized saliency in [0, 1]
    """
    # Clamp to non-negative
    saliency = saliency.clamp(min=0)
    
    # Percentile clipping
    if percentile < 100:
        threshold = torch.quantile(saliency.flatten(), percentile / 100.0)
        saliency = saliency.clamp(max=threshold)
    
    # Normalize to [0, 1]
    smin, smax = saliency.min(), saliency.max()
    if smax > smin:
        saliency = (saliency - smin) / (smax - smin)
    else:
        saliency = torch.zeros_like(saliency)
    
    return saliency


def create_saliency_overlay(
    image: np.ndarray,
    actor_saliency: np.ndarray,
    critic_saliency: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'hot'
) -> np.ndarray:
    """Create visualization overlay combining image with saliency maps.
    
    Creates a side-by-side view: [original | actor saliency | critic saliency]
    
    Args:
        image: Original image (H, W, C) in uint8 [0, 255]
        actor_saliency: Actor saliency (H, W) in [0, 1]
        critic_saliency: Critic saliency (H, W) in [0, 1]
        alpha: Blending factor for overlay
        colormap: Colormap name for saliency ('hot', 'jet', 'viridis')
        
    Returns:
        Combined visualization (H, 3*W, C) in uint8
    """
    import matplotlib.pyplot as plt
    
    H, W, C = image.shape
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap to saliency maps (returns RGBA)
    actor_colored = (cmap(actor_saliency)[..., :3] * 255).astype(np.uint8)
    critic_colored = (cmap(critic_saliency)[..., :3] * 255).astype(np.uint8)
    
    # Blend with original image
    image_float = image.astype(np.float32)
    actor_overlay = (1 - alpha) * image_float + alpha * actor_colored.astype(np.float32)
    critic_overlay = (1 - alpha) * image_float + alpha * critic_colored.astype(np.float32)
    
    actor_overlay = np.clip(actor_overlay, 0, 255).astype(np.uint8)
    critic_overlay = np.clip(critic_overlay, 0, 255).astype(np.uint8)
    
    # Concatenate horizontally
    combined = np.concatenate([image, actor_overlay, critic_overlay], axis=1)
    
    return combined


@torch.no_grad()
def compute_episode_saliency(
    world_model,
    actor_critic,
    episode_cache: torch.Tensor,
    preprocess_fn,
    stride: int = 5,
    mask_sigma: float = 5.0,
    blur_sigma: float = 3.0,
    act_discrete: bool = False,
    saliency_every_n: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute saliency maps for an entire evaluation episode.
    
    Args:
        world_model: World model module
        actor_critic: Actor-critic module
        episode_cache: Stacked episode data (T, ...) with 'image', 'action', 'is_first'
        preprocess_fn: Preprocessing function
        stride: Saliency grid stride
        mask_sigma: Gaussian mask sigma
        blur_sigma: Blur sigma
        act_discrete: Whether actions are discrete
        saliency_every_n: Compute saliency every N frames (1 = all frames)
        
    Returns:
        actor_saliency_maps: (T, H, W) tensor of actor saliency
        critic_saliency_maps: (T, H, W) tensor of critic saliency
    """
    T = episode_cache.shape[0]
    device = episode_cache['image'].device
    
    # Get image dimensions
    sample_image = episode_cache['image'][0]
    if sample_image.max() > 1.0:
        H, W, C = sample_image.shape
    else:
        H, W, C = sample_image.shape
    
    # Initialize saliency storage
    actor_saliency_list = []
    critic_saliency_list = []
    
    # Initialize state
    stoch, deter = world_model.dynamics.initial(1)
    prev_action = torch.zeros(1, episode_cache['action'].shape[-1], device=device)
    
    for t in range(T):
        # Get current observation
        obs = {k: v[t:t+1].squeeze(0) if v.dim() > 0 else v[t] for k, v in episode_cache.items()}
        is_first = obs.get('is_first', torch.tensor(False, device=device))
        
        # Reset state if new episode
        if is_first.item() if is_first.dim() == 0 else is_first[0].item():
            stoch, deter = world_model.dynamics.initial(1)
            prev_action = torch.zeros(1, episode_cache['action'].shape[-1], device=device)
        
        if t % saliency_every_n == 0:
            # Compute saliency for this frame
            actor_sal, critic_sal = compute_saliency_step(
                world_model, actor_critic,
                obs, stoch.squeeze(0), deter.squeeze(0), prev_action.squeeze(0),
                is_first, preprocess_fn,
                stride, mask_sigma, blur_sigma, act_discrete
            )
            
            # Upsample to full resolution
            actor_sal = upsample_saliency(actor_sal, (H, W))
            critic_sal = upsample_saliency(critic_sal, (H, W))
            
            # Normalize
            actor_sal = normalize_saliency(actor_sal)
            critic_sal = normalize_saliency(critic_sal)
        else:
            # Use previous saliency (or zeros for first frame)
            if len(actor_saliency_list) > 0:
                actor_sal = actor_saliency_list[-1]
                critic_sal = critic_saliency_list[-1]
            else:
                actor_sal = torch.zeros(H, W, device=device)
                critic_sal = torch.zeros(H, W, device=device)
        
        actor_saliency_list.append(actor_sal)
        critic_saliency_list.append(critic_sal)
        
        # Update state for next step (need to do observation step)
        p_obs = preprocess_fn({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() < 4 else v 
                               for k, v in obs.items()})
        embed = world_model.encoder(p_obs)
        stoch, deter, _ = world_model.dynamics.obs_step(
            stoch, deter, prev_action, embed, 
            is_first.unsqueeze(0) if is_first.dim() == 0 else is_first.unsqueeze(0)
        )
        prev_action = obs['action'].unsqueeze(0) if 'action' in obs else prev_action
    
    actor_saliency_maps = torch.stack(actor_saliency_list, dim=0)  # (T, H, W)
    critic_saliency_maps = torch.stack(critic_saliency_list, dim=0)
    
    return actor_saliency_maps, critic_saliency_maps

