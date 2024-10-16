"""
train2, but replacing UNet with DiT

Main components:

1. Forward Diffusion Process: Adds progressively more Gaussian noise to the data.
2. Reverse Process: A transformer-based model predicts and removes noise step by step.
3. Loss Function: Minimizes the difference between actual and predicted noise.
4. Noise Schedule: Defines how much noise to add at each timestep.
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid

from .transformer import DiT  # Import the DiT model
from .ddpm import ddpm_schedules

import torch.distributed as dist
import os

def forward_diffusion(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_t = (
        noise_schedule["sqrtab"][_ts] * x_0
        + noise_schedule["sqrtmab"][_ts] * noise
    )
    return x_t, noise


def clamp(x):
    return x.clamp(-1, 1)


def sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device, clamp_sample=True):
    x_i = x_T
    for i in range(n_T, 0, -1):
        z = torch.randn_like(x_i) if i > 1 else 0
        t = torch.full((x_i.shape[0],), i / n_T, device=device)
        predicted_noise = denoising_model(x_i, t)
        x_i = (
            noise_schedule["oneover_sqrta"][i] * (x_i - predicted_noise * noise_schedule["mab_over_sqrtmab"][i])
            + noise_schedule["sqrt_beta_t"][i] * z
        )
        if clamp_sample:
            x_i = clamp(x_i)
    return x_i


def train_loop(denoising_model, dataloader, optimizer, noise_schedule, n_T, total_steps, device, log_every=10, sample_every=100, clamp_sample=True):
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
            
            t = torch.randint(1, n_T + 1, (x.shape[0],)).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule)
            
            predicted_noise = denoising_model(x_t, t / n_T, y)[:, :3]  # Use only the first 3 channels
            loss = criterion(predicted_noise, true_noise)
            
            loss.backward()
            optimizer.step()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            if step % log_every == 0:
                print(f"Step {step}/{total_steps}, Loss: {loss_ema:.4f}")
            
            if step % sample_every == 0:
                denoising_model.eval()
                with torch.no_grad():
                    n_sample = 8
                    x_T = torch.randn(n_sample, 3, 32, 32).to(device)
                    sampled_images = sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device, clamp_sample=clamp_sample)
                    
                    xset = torch.cat([sampled_images, x[:n_sample]], dim=0)
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                    img_output_dir = Path("data/image_gen/nano_diffusion/train3")
                    img_output_dir.mkdir(parents=True, exist_ok=True)
                    save_image(grid, img_output_dir / f"ddpm_sample_cifar_step{step}.png")
                denoising_model.train()
            
            step += 1


def main():
    # Hyperparameters
    n_T = 1000
    total_steps = 100000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_every = 20
    sample_every = 500
    clamp_sample = False

    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Initialize the process group
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # Multi-GPU setup
            raise NotImplementedError("multi-node not supported")
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        else:
            # Single GPU setup
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
    else:
        # CPU setup
        dist.init_process_group(backend='nccl', rank=0, world_size=1)

    # Define the denoising model (DiT)
    denoising_model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        num_classes=10,
        learn_sigma=True,
        hidden_size=64,
        mlp_ratio=2,
        depth=3,
        num_heads=6,
        class_dropout_prob=0.1,
    ).to(device)
    
    # Define the optimizer
    optimizer = optim.AdamW(denoising_model.parameters(), lr=1e-4, weight_decay=0.05)

    # Define the noise schedule
    noise_schedule = ddpm_schedules(1e-4, 0.02, n_T)
    for k, v in noise_schedule.items():
        noise_schedule[k] = v.to(device)

    # Train the model
    train_loop(
        denoising_model, dataloader, optimizer, noise_schedule,
        n_T, total_steps, device, log_every, sample_every, clamp_sample=clamp_sample
    )

if __name__ == "__main__":
    main()
