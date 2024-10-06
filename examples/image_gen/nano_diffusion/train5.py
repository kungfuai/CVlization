"""
diffuser-based training pipeline

see examples/image_gen/diffuser_unconditional

Modified on top of train2, but bringing in ingredients from diffuser-based training pipeline.

"""

import math
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid
from cvlization.torch.training_pipeline.image_gen.dit import DiT
from cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.pipeline import UNet2DModel

import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import argparse
try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")
from .ema import create_ema_model
from cleanfid import fid
import tempfile


def forward_diffusion_old(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_t = (
        noise_schedule["sqrtab"][_ts] * x_0
        + noise_schedule["sqrtmab"][_ts] * noise
    )
    return x_t, noise


def forward_diffusion(x_0, t, noise_schedule, noise=None):
    _ts = t.view(-1, 1, 1, 1)
    if noise is None:
        noise = torch.randn_like(x_0)
    assert _ts.max() < len(noise_schedule["alphas_cumprod"]), f"t={_ts.max()} is larger than the length of noise_schedule: {len(noise_schedule['alphas_cumprod'])}"
    alpha_prod_t = noise_schedule["alphas_cumprod"][_ts]
    x_t = (alpha_prod_t ** 0.5) * x_0 + ((1 - alpha_prod_t) ** 0.5) * noise
    return x_t, noise


def threshold_sample(sample: torch.Tensor, dynamic_thresholding_ratio=0.995, sample_max_value=1.0) -> torch.Tensor:
    """
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    photorealism as well as better image-text alignment, especially when using very large guidance weights."

    https://arxiv.org/abs/2205.11487
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape

    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

    # Flatten sample for doing quantile calculation along each image
    sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

    abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(
        s, min=1, max=sample_max_value
    )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
    s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
    sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)

    return sample


def sample_by_denoising_old(denoising_model, x_T, noise_schedule, n_T, device):
    x_i = x_T.to(device)
    for i in range(n_T, 0, -1):
        z = torch.randn_like(x_i) if i > 1 else 0
        t = torch.full((x_i.shape[0],), i / n_T, device=device)
        predicted_noise = denoising_model(x_i, t)
        x_i = (
            noise_schedule["oneover_sqrta"][i] * (x_i - predicted_noise * noise_schedule["mab_over_sqrtmab"][i])
            + noise_schedule["sqrt_beta_t"][i] * z
        )
    x_i = (x_i / 2 + 0.5).clamp(0, 1)
    return x_i


def denoising_step(denoising_model, x_t, t, noise_schedule, thresholding=False, clip_sample=True, clip_sample_range=1.0):
    """
    This is the backward diffusion step, with the effect of denoising.
    """
    if isinstance(t, int):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    else:
        t_tensor = t
    model_output = denoising_model(x_t, t_tensor)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample
    
    # Extract relevant values from noise_schedule
    alpha_prod_t = noise_schedule["alphas_cumprod"][t_tensor]
    alpha_prod_t.to(x_t.device)
    # deal with t=0 case where t can be a tensor
    alpha_prod_t_prev = torch.where(t_tensor > 0, 
                                    noise_schedule["alphas_cumprod"][t_tensor - 1], 
                                    torch.ones_like(t_tensor, device=x_t.device))
    alpha_prod_t_prev.to(alpha_prod_t.device)

    # Reshape alpha_prod_t_prev for proper broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Compute the previous sample mean
    pred_original_sample = (x_t - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    # print("x_t mean:", x_t.mean().item())
    # print("t:", t)
    # print("model_output mean:", model_output.mean().item())
    # print("pred_original_sample mean (before clipping):", pred_original_sample.mean().item())

    if clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -clip_sample_range, clip_sample_range)

    # Compute the coefficients for pred_original_sample and current sample
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

    # print(f"pred_original_sample shape: {pred_original_sample.shape}")
    # print("alpha_prod_t_prev", alpha_prod_t_prev.shape)
    # print("current_beta_t", current_beta_t.shape)
    # print("beta_prod_t", beta_prod_t.shape)
    # print("alpha_prod_t", alpha_prod_t.shape)
    # print(f"pred_original_sample_coeff shape: {pred_original_sample_coeff.shape}")
    # print(f"current_sample_coeff shape: {current_sample_coeff.shape}")

    # Compute the previous sample
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t

    # Add noise
    variance = torch.zeros_like(x_t)
    variance_noise = torch.randn_like(x_t)
    
    # Handle t=0 case where t can be a tensor
    non_zero_mask = (t != 0).float().view(-1, 1, 1, 1)
    variance = non_zero_mask * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t)
    variance = torch.clamp(variance, min=1e-20)

    pred_prev_sample = pred_prev_sample + (variance ** 0.5) * variance_noise

    if thresholding:
        pred_prev_sample = threshold_sample(pred_prev_sample)
    
    # for debug, print out intermediate values
    # print("alpha_prod_t", alpha_prod_t[0].item())
    # print("alpha_prod_t_prev", alpha_prod_t_prev[0].item())
    # print("beta_prod_t", beta_prod_t[0].item())
    # print("beta_prod_t_prev", beta_prod_t_prev[0].item())
    # print("current_alpha_t", current_alpha_t[0].item())
    # print("current_beta_t", current_beta_t[0].item())
    # print("pred_original_sample mean", pred_original_sample.mean().item())
    # print("pred_original_sample_coeff", pred_original_sample_coeff[0].item())
    # print("current_sample_coeff", current_sample_coeff[0].item())

    return pred_prev_sample


def sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device, thresholding=False, clip_sample=True, clip_sample_range=1.0, seed=0):
    torch.manual_seed(seed)

    x_t = x_T.to(device)
    for t in range(n_T - 1, -1, -1):
        # t_tensor = torch.full((x_t.shape[0],), t / n_T, device=device)
        x_t = denoising_step(denoising_model, x_t, t, noise_schedule, thresholding, clip_sample, clip_sample_range)

    # print("raw x_t range", x_t.min(), x_t.max())
    x_t = (x_t / 2 + 0.5).clamp(0, 1)
    # print("after clamp", x_t.min(), x_t.max())
    return x_t


def compute_validation_loss(model, val_dataloader, noise_schedule, n_T, device, criterion, use_loss_mean=True, ema_model=None):
    model.eval()
    if ema_model is not None:
        ema_model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule)
            
            # Compute loss for the main model
            predicted_noise = model(x_t, t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample
            
            loss = criterion(predicted_noise, true_noise)
            if use_loss_mean:
                loss = loss.mean()
            
            total_loss += loss.item()
            
            # Compute loss for the EMA model if provided
            if ema_model is not None:
                ema_predicted_noise = ema_model(x_t, t)
                if hasattr(ema_predicted_noise, "sample"):
                    ema_predicted_noise = ema_predicted_noise.sample
                
                ema_loss = criterion(ema_predicted_noise, true_noise)
                if use_loss_mean:
                    ema_loss = ema_loss.mean()
                
                total_ema_loss += ema_loss.item()
            
            num_batches += 1
    
    model.train()
    if ema_model is not None:
        return total_loss / num_batches, total_ema_loss / num_batches
    else:
        return total_loss / num_batches


def denoise_and_compare(model, images, noise_schedule, n_T, device):
    torch.manual_seed(10)
    model.eval()
    with torch.no_grad():
        # Add noise to the images
        t = torch.randint(0, n_T, (images.shape[0],), device=device)
        x_t, _ = forward_diffusion(images, t, noise_schedule)
        
        # Denoise the images
        pred_noise = model(x_t, t)
        if hasattr(pred_noise, "sample"):
            pred_noise = pred_noise.sample
        pred_previous_images = denoising_step(model, x_t, t, noise_schedule)
        pred_original_images = (x_t - pred_noise * noise_schedule["sqrt_beta_t"][t]) / noise_schedule["oneover_sqrta"][t]
    model.train()
    return pred_previous_images, pred_original_images


def save_comparison_grid(original, denoised, ema_denoised, step, prefix, output_dir):
    # Combine images into a grid
    grid_images = torch.cat([original, denoised], dim=0)
    if ema_denoised is not None:
        grid_images = torch.cat([grid_images, ema_denoised], dim=0)
    
    grid = make_grid(grid_images, nrow=original.shape[0], normalize=True, scale_each=True)
    save_image(grid, output_dir / f"{prefix}_comparison_step{step}.png")


def compute_fid(real_images, generated_images, device='cuda'):
    """Compute FID between real and generated images using clean-fid."""
    with tempfile.TemporaryDirectory() as temp_dir:
        real_path = Path(temp_dir) / 'real'
        gen_path = Path(temp_dir) / 'gen'
        real_path.mkdir()
        gen_path.mkdir()

        # Convert images to numpy arrays and save them
        for i, img in enumerate(real_images):
            img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            np.save(real_path / f'{i}.npy', img_np)
        for i, img in enumerate(generated_images):
            img_np = (img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            np.save(gen_path / f'{i}.npy', img_np)

        # Compute FID
        score = fid.compute_fid(str(real_path), str(gen_path), device=device, mode="clean")

    return score


def train_loop(denoising_model, train_dataloader, val_dataloader, optimizer, lr_scheduler, noise_schedule, n_T, total_steps, device, log_every=10, sample_every=100, save_every=5000, validate_every=1000, fid_every=10000, logger="none", max_grad_norm=1.0, use_loss_mean=True, use_ema=False, ema_beta=0.9999):
    criterion = MSELoss()
    denoising_model.train()
    
    # Initialize EMA model if specified
    ema_model = None
    if use_ema:
        ema_model, ema = create_ema_model(denoising_model, ema_beta)
        ema_model = ema_model.to(device)
        ema_model.eval()
    
    # Get a batch of training and validation images for denoising comparison
    train_images_for_denoising, _ = next(iter(train_dataloader))
    train_images_for_denoising = train_images_for_denoising[:8].to(device)
    val_images_for_denoising, _ = next(iter(val_dataloader))
    val_images_for_denoising = val_images_for_denoising[:8].to(device)

    img_output_dir = Path("data/image_gen/nano_diffusion/train5")
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # Log true samples only once at the beginning
    if logger == "wandb":
        wandb.log({
            "true_train_samples": [wandb.Image(img) for img in train_images_for_denoising],
            "true_val_samples": [wandb.Image(img) for img in val_images_for_denoising],
        })

    # Get a larger batch of real images for FID computation
    real_images_for_fid, _ = next(iter(DataLoader(train_dataloader.dataset, batch_size=1000, shuffle=True)))
    real_images_for_fid = real_images_for_fid.to(device)

    step = 0
    
    while step < total_steps:
        for x, y in train_dataloader:
            if step >= total_steps:
                break
            
            optimizer.zero_grad()
            x = x.to(device)
            
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule)
            
            predicted_noise = denoising_model(x_t, t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample
            loss = criterion(predicted_noise, true_noise)
            if use_loss_mean:
                loss = loss.mean()
            
            loss.backward()
            
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            
            # Update EMA model if specified
            if use_ema:
                ema.step_ema(ema_model, denoising_model)
            
            if step % log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}/{total_steps}, Train Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                if logger == "wandb":
                    wandb.log({
                        "step": step,
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                    })
            
            if step % validate_every == 0:
                val_loss = compute_validation_loss(denoising_model, val_dataloader, noise_schedule, n_T, device, criterion, use_loss_mean)
                
                # Generate denoised images for training and validation sets
                _, denoised_train = denoise_and_compare(denoising_model, train_images_for_denoising, noise_schedule, n_T, device)
                _, denoised_val = denoise_and_compare(denoising_model, val_images_for_denoising, noise_schedule, n_T, device)
                
                ema_denoised_train = None
                ema_denoised_val = None
                if use_ema:
                    ema_val_loss = compute_validation_loss(ema_model, val_dataloader, noise_schedule, n_T, device, criterion, use_loss_mean)
                    _, ema_denoised_train = denoise_and_compare(ema_model, train_images_for_denoising, noise_schedule, n_T, device)
                    _, ema_denoised_val = denoise_and_compare(ema_model, val_images_for_denoising, noise_schedule, n_T, device)
                
                # Save comparison grids
                # save_comparison_grid(train_images_for_denoising, denoised_train, ema_denoised_train, step, "train", img_output_dir)
                # save_comparison_grid(val_images_for_denoising, denoised_val, ema_denoised_val, step, "val", img_output_dir)
                
                print(f"Step {step}/{total_steps}, Validation Loss: {val_loss:.4f}")
                if logger == "wandb":
                    log_dict = {
                        "step": step,
                        "val_loss": val_loss,
                        "noisy_train_samples": [wandb.Image(img) for img in train_images_for_denoising],
                        "noisy_val_samples": [wandb.Image(img) for img in val_images_for_denoising],
                        "denoised_train_samples": [wandb.Image(img) for img in denoised_train],
                        "denoised_val_samples": [wandb.Image(img) for img in denoised_val],
                    }
                    if use_ema:
                        log_dict.update({
                            "ema_val_loss": ema_val_loss,
                            "ema_denoised_train_samples": [wandb.Image(img) for img in ema_denoised_train],
                            "ema_denoised_val_samples": [wandb.Image(img) for img in ema_denoised_val],
                        })
                    wandb.log(log_dict)
            
            if step % sample_every == 0:
                denoising_model.eval()
                with torch.no_grad():
                    n_sample = 8
                    x_T = torch.randn(n_sample, 3, 32, 32).to(device)
                    sampled_images = sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device)
                    xset = torch.cat([sampled_images, x[:n_sample]], dim=0)
                    grid = make_grid(xset, normalize=True, nrow=4, scale_each=True)
                    img_output_dir = Path("data/image_gen/nano_diffusion/train5")
                    img_output_dir.mkdir(parents=True, exist_ok=True)
                    save_image(grid, img_output_dir / f"ddpm_sample_cifar_step{step}.png")
                    
                    if logger == "wandb":
                        # save individual images
                        images_processed = (sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
                        wandb.log({
                            "test_samples": [wandb.Image(img) for img in images_processed],
                        })
                    if use_ema:
                        ema_sampled_images = sample_by_denoising(ema_model, x_T, noise_schedule, n_T, device)
                        ema_images_processed = (ema_sampled_images * 255).permute(0, 2, 3, 1).cpu().numpy().round().astype("uint8")
                        if logger == "wandb":
                            wandb.log({
                                "ema_test_samples": [wandb.Image(img) for img in ema_images_processed],
                            })
                denoising_model.train()
            
            if step % save_every == 0 and step > 0:
                checkpoint_path = f"model_checkpoint_step_{step}.pth"
                torch.save(denoising_model.state_dict(), checkpoint_path)
                print(f"Model saved at step {step}")
                if logger == "wandb":
                    wandb.save(checkpoint_path)
                if use_ema:
                    ema_checkpoint_path = f"ema_model_checkpoint_step_{step}.pth"
                    torch.save(ema_model.state_dict(), ema_checkpoint_path)
                    print(f"EMA Model saved at step {step}")
                    if logger == "wandb":
                        wandb.save(ema_checkpoint_path)
            
            if step % fid_every == 0 and step > 0:
                denoising_model.eval()
                with torch.no_grad():
                    n_sample = 1000  # Generate the same number of images as real_images_for_fid
                    x_T = torch.randn(n_sample, 3, 32, 32).to(device)
                    generated_images = sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device)
                    
                    fid_score = compute_fid(real_images_for_fid, generated_images, device=device)
                    print(f"Step {step}/{total_steps}, FID: {fid_score:.4f}")
                    
                    if logger == "wandb":
                        wandb.log({
                            "step": step,
                            "fid": fid_score,
                        })
                    
                    if use_ema:
                        ema_generated_images = sample_by_denoising(ema_model, x_T, noise_schedule, n_T, device)
                        ema_fid_score = compute_fid(real_images_for_fid, ema_generated_images, device=device)
                        print(f"Step {step}/{total_steps}, EMA FID: {ema_fid_score:.4f}")
                        
                        if logger == "wandb":
                            wandb.log({
                                "step": step,
                                "ema_fid": ema_fid_score,
                            })
            
            step += 1
    
    # Save final model
    final_model_path = "final_model.pth"
    torch.save(denoising_model.state_dict(), final_model_path)
    print("Final model saved")
    if logger == "wandb":
        wandb.save(final_model_path)
    if use_ema:
        final_ema_model_path = "final_ema_model.pth"
        torch.save(ema_model.state_dict(), final_ema_model_path)
        print("Final EMA model saved")
        if logger == "wandb":
            wandb.save(final_ema_model_path)


def create_model(net: str = "unet", resolution: int = 32, in_channels: int = 3):
    if net == "dit_t0":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=3,
            num_heads=1,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t1":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32 * 6,
            mlp_ratio=2,
            depth=3,
            num_heads=6,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t2":
        return DiT(
            input_size=32,
            patch_size=2,
            in_channels=in_channels,
            learn_sigma=False,
            hidden_size=32,
            mlp_ratio=2,
            depth=12,
            num_heads=1,
            class_dropout_prob=0.1,
        )
    elif net == "dit_t3":
        model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=in_channels,
                learn_sigma=False,
                hidden_size=32 * 6,
                mlp_ratio=2,
                depth=12,
                num_heads=6,
                class_dropout_prob=0.1,
            )
    elif net == "dit_s2":
        model = DiT(
                depth=12,
                in_channels=in_channels,
                hidden_size=384,
                patch_size=2,
                num_heads=6,
                learn_sigma=False,
            )
    elif net == "dit_b2":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=2,
            num_heads=6,
            learn_sigma=False,
        )
    elif net == "dit_b4":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=384,
            patch_size=4,
            num_heads=6,
            learn_sigma=False,
        )
    elif net == "dit_l2":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=2,
            num_heads=12,
            learn_sigma=False,
        )
    elif net == "dit_l4":
        model = DiT(
            depth=12,
            in_channels=in_channels,
            hidden_size=768,
            patch_size=4,
            num_heads=12,
            learn_sigma=False,
        )
    elif net == "unet":
        model = UNet2DModel(
            sample_size=resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        raise ValueError(f"Unsupported network architecture: {net}")
    
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser(description="DDPM training for CIFAR10")
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none", help="Logging method")
    parser.add_argument("--net", type=str, choices=[
        "dit_t0", "dit_t1", "dit_t2", "dit_t3",
        "dit_s2", "dit_b2", "dit_b4", "unet",
        "dit_b2", "dit_b4", "dit_l2", "dit_l4",
    ], default="unet", help="Network architecture")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--resolution", type=int, default=32, help="Resolution of the image. Only used for unet.")
    
    # Add config parameters to argparser
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps in the diffusion process")
    parser.add_argument("--total_steps", type=int, default=120000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--lr_min", type=float, default=2e-6, help="Minimum learning rate")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--sample_every", type=int, default=1500, help="Sample every N steps")
    parser.add_argument("--save_every", type=int, default=60000, help="Save model every N steps")
    parser.add_argument("--validate_every", type=int, default=1500, help="Compute validation loss every N steps")
    parser.add_argument("--fid_every", type=int, default=1500, help="Compute FID every N steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--max_grad_norm", type=float, default=-1, help="Maximum norm for gradient clipping")
    parser.add_argument("--use_loss_mean", action="store_true", help="Use loss.mean() instead of just loss")
    parser.add_argument("--watch_model", action="store_true", help="Use wandb to watch the model")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average (EMA) for the model")
    parser.add_argument("--ema_beta", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--random_flip", action="store_true", help="Randomly flip images horizontally")
    
    args = parser.parse_args()

    # Use args to create config
    config = {
        "num_timesteps": args.num_timesteps,
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_min": args.lr_min,
        "log_every": args.log_every,
        "sample_every": args.sample_every,
        "save_every": args.save_every,
        "validate_every": args.validate_every,
        "net": args.net,
        "in_channels": args.in_channels,
        "resolution": args.resolution,
        "max_grad_norm": args.max_grad_norm,
        "use_loss_mean": args.use_loss_mean,
        "watch_model": args.watch_model,
        "use_ema": args.use_ema,
        "ema_beta": args.ema_beta,
    }

    # Initialize wandb if specified
    if args.logger == "wandb":
        import wandb
        wandb.init(project=os.getenv("WANDB_PROJECT") or "Diffuser Unconditional", config=config)
        config = wandb.config
    else:
        from types import SimpleNamespace
        config = SimpleNamespace(**config)

    # Use config
    num_timesteps = config.num_timesteps
    total_steps = config.total_steps
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if args.random_flip else lambda x: x,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Initialize the process group
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # Multi-GPU setup
            raise NotImplementedError("multi-node not supported")
        
    # Define the denoising model
    denoising_model = create_model(args.net, 32, 3).to(device)
    
    # Add wandb.watch here, controlled by the new argument
    if args.logger == "wandb" and config.watch_model:
        wandb.watch(denoising_model, log="all", log_freq=100)

    # Define the optimizer
    optimizer = optim.AdamW(denoising_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Define the learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    # Define the noise schedule
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noise_schedule = {
        "betas": betas.to(device),
        "alphas": alphas.to(device),
        "alphas_cumprod": alphas_cumprod.to(device),
    }
    for k, v in noise_schedule.items():
        print(k, v.shape)

    # Train the model
    train_loop(
        denoising_model, train_dataloader, val_dataloader, optimizer, lr_scheduler, noise_schedule,
        num_timesteps,
        total_steps=args.total_steps,
        device=args.device,
        log_every=args.log_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        validate_every=args.validate_every,
        logger=args.logger,
        max_grad_norm=config.max_grad_norm,
        use_loss_mean=config.use_loss_mean,
        use_ema=config.use_ema,
        ema_beta=config.ema_beta,
        fid_every=args.fid_every,
    )

    # Close wandb run if it was used
    if args.logger == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()