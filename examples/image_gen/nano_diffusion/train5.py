"""
diffuser-based training pipeline

see examples/image_gen/diffuser_unconditional

Modified on top of train2, but bringing in ingredients from diffuser-based training pipeline.

"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid
# Import the DiT model
from .ddpm import ddpm_schedules
from cvlization.torch.training_pipeline.dit.dit import DiT
from cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.pipeline import UNet2DModel

import torch.distributed as dist
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
try:
    import wandb
except ImportError:
    print("wandb not installed, skipping")


def forward_diffusion_old(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_t = (
        noise_schedule["sqrtab"][_ts] * x_0
        + noise_schedule["sqrtmab"][_ts] * noise
    )
    return x_t, noise


def forward_diffusion(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
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


def denoising_step(denoising_model, x_t, t, noise_schedule, thresholding=False, clip_sample=True, clip_sample_range=1.0, prediction_type="epsilon"):
    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device)
    model_output = denoising_model(x_t, t_tensor)
    if hasattr(model_output, "sample"):
        model_output = model_output.sample
    
    # Extract relevant values from noise_schedule
    alpha_prod_t = noise_schedule["alphas_cumprod"][t_tensor]
    alpha_prod_t.to(x_t.device)
    alpha_prod_t_prev = noise_schedule["alphas_cumprod"][t_tensor - 1] if t > 0 else torch.tensor(1.0).to(x_t.device)
    alpha_prod_t_prev.to(alpha_prod_t.device)

    # Reshape alpha_prod_t_prev for proper broadcasting
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Compute the previous sample mean
    if prediction_type == "epsilon":
        pred_original_sample = (x_t - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    elif prediction_type == "sample":
        pred_original_sample = model_output
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

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
    variance = 0
    variance_noise = torch.randn_like(x_t)
    if t > 0:
        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t)
        variance = torch.clamp(variance, min=1e-20)

    pred_prev_sample = pred_prev_sample + (variance ** 0.5) * variance_noise

    if thresholding:
        pred_prev_sample = threshold_sample(pred_prev_sample)

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



def train_loop(denoising_model, dataloader, optimizer, lr_scheduler, noise_schedule, n_T, total_steps, device, log_every=10, sample_every=100, logger="none"):
    criterion = MSELoss()
    denoising_model.train()
    
    loss_ema = None
    step = 0
    
    while step < total_steps:
        for x, y in dataloader:
            if step >= total_steps:
                break
            
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            
            t = torch.randint(0, n_T, (x.shape[0],)).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule)
            
            predicted_noise = denoising_model(x_t, t)
            if hasattr(predicted_noise, "sample"):
                predicted_noise = predicted_noise.sample
            loss = criterion(predicted_noise, true_noise)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            if step % log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step}/{total_steps}, Loss: {loss_ema:.4f}, LR: {current_lr:.6f}")
                if logger == "wandb":
                    wandb.log({
                        "step": step,
                        "loss": loss_ema,
                        "learning_rate": current_lr,
                    })
            
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
                        wandb.log({
                            "generated_images": wandb.Image(grid),
                        })
                denoising_model.train()
            
            step += 1
        
        # Save model checkpoint
        if step % (total_steps // 10) == 0:
            checkpoint_path = f"model_checkpoint_step_{step}.pth"
            torch.save(denoising_model.state_dict(), checkpoint_path)
            if logger == "wandb":
                wandb.save(checkpoint_path)

    # Save final model
    final_model_path = "final_model.pth"
    torch.save(denoising_model.state_dict(), final_model_path)
    if logger == "wandb":
        wandb.save(final_model_path)


def main():
    parser = argparse.ArgumentParser(description="DDPM training for CIFAR10")
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none", help="Logging method")
    args = parser.parse_args()

    # Hyperparameters
    config = {
        "n_T": 1000,
        "total_steps": 50000,
        "batch_size": 128,
        "learning_rate": 7e-5,
        "weight_decay": 1e-6,
        "lr_min": 5e-6,
        "log_every": 20,
        "sample_every": 100,
    }

    # Initialize wandb if specified
    if args.logger == "wandb":
        import wandb
        wandb.init(project="ddpm-cifar10", config=config)
        config = wandb.config
    else:
        from types import SimpleNamespace
        config = SimpleNamespace(**config)

    # Use config
    n_T = config.n_T
    total_steps = config.total_steps
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_every = config.log_every
    sample_every = config.sample_every

    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # Initialize the process group
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # Multi-GPU setup
            raise NotImplementedError("multi-node not supported")
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        
    # Define the denoising model (DiT)
    # denoising_model = DiT(
    #     depth=12,
    #     input_size=32,
    #     in_channels=3,
    #     hidden_size=384,
    #     patch_size=2,
    #     num_heads=6,
    #     learn_sigma=False,
    # ).to(device)
    # denoising_model = DiT(
    #     input_size=32,
    #     patch_size=2,
    #     in_channels=3,
    #     learn_sigma=False,
    #     hidden_size=32 * 6,
    #     mlp_ratio=4,
    #     depth=12,
    #     num_heads=6,
    #     class_dropout_prob=0.1,
    # ).to(device)

    denoising_model = UNet2DModel(
        sample_size=32,
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
    ).to(device)
    
    # Define the optimizer
    optimizer = optim.AdamW(denoising_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Define the learning rate scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.lr_min)

    # Define the noise schedule
    betas = torch.linspace(1e-4, 0.02, n_T)
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
        denoising_model, dataloader, optimizer, lr_scheduler, noise_schedule,
        n_T, total_steps, device, log_every, sample_every, logger=args.logger
    )

    # Close wandb run if it was used
    if args.logger == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()