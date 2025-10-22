"""
train 1, plus a simple UNet

Main components:

1. Forward Diffusion Process: Adds progressively more Gaussian noise to the data.
2. Reverse Process: A neural network predicts and removes noise step by step.
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
from tqdm import tqdm

from .unet import NaiveUnet
from .ddpm import ddpm_schedules


def forward_diffusion(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_t = (
        noise_schedule["sqrtab"][_ts] * x_0
        + noise_schedule["sqrtmab"][_ts] * noise
    )
    return x_t, noise


def sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device):
    x_i = x_T
    for i in range(n_T, 0, -1):
        z = torch.randn_like(x_i) if i > 1 else 0
        t = torch.full((x_i.shape[0],), i / n_T, device=device)
        predicted_noise = denoising_model(x_i, t)
        x_i = (
            noise_schedule["oneover_sqrta"][i] * (x_i - predicted_noise * noise_schedule["mab_over_sqrtmab"][i])
            + noise_schedule["sqrt_beta_t"][i] * z
        )
    return x_i


def train_loop(denoising_model, dataloader, optimizer, noise_schedule, n_T, total_steps, device, log_every=10, sample_every=100):
    criterion = MSELoss()
    denoising_model.train()
    
    loss_ema = None
    step = 0
    
    while step < total_steps:
        for x, _ in dataloader:
            if step >= total_steps:
                break
            
            optimizer.zero_grad()
            x = x.to(device)
            
            t = torch.randint(1, n_T + 1, (x.shape[0],)).to(device)
            x_t, true_noise = forward_diffusion(x, t, noise_schedule)
            
            predicted_noise = denoising_model(x_t, t / n_T)
            loss = criterion(predicted_noise, true_noise)
            
            loss.backward()
            optimizer.step()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            
            if step % log_every == 0:
                print(f"Step {step + 1}/{total_steps}, Loss: {loss_ema:.4f}")
            
            if step % sample_every == 0:
                denoising_model.eval()
                with torch.no_grad():
                    n_sample = 8
                    x_T = torch.randn(n_sample, 3, 32, 32).to(device)
                    sampled_images = sample_by_denoising(denoising_model, x_T, noise_schedule, n_T, device)
                    
                    xset = torch.cat([sampled_images, x[:n_sample]], dim=0)
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                    img_output_dir = Path("data/image_gen/nano_diffusion/train2")
                    img_output_dir.mkdir(parents=True, exist_ok=True)
                    save_image(grid, img_output_dir / f"ddpm_sample_cifar_step{step+1}.png")
                denoising_model.train()
            
            step += 1


def main():
    # Hyperparameters
    n_T = 1000
    total_steps = 100000  # Replace num_epochs with total_steps
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_every = 20
    sample_every = 500

    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    # Define the denoising model (NaiveUnet)
    denoising_model = NaiveUnet(in_channels=3, out_channels=3, n_feat=128).to(device)
    
    # Define the optimizer
    optimizer = optim.Adam(denoising_model.parameters(), lr=1e-5)

    # Define the noise schedule
    noise_schedule = ddpm_schedules(1e-4, 0.02, n_T)
    for k, v in noise_schedule.items():
        noise_schedule[k] = v.to(device)

    # Train the model
    train_loop(
        denoising_model, dataloader, optimizer, noise_schedule,
        n_T, total_steps, device, log_every, sample_every
    )

if __name__ == "__main__":
    main()