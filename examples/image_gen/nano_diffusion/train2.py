"""

## Main components

1. Forward Diffusion Process:

Adds progressively more Gaussian noise to the data according to a noise schedule β.

2. Reverse Process:

A neural network is trained to predict the noise at each time step and is used to denoise the data in the reverse direction, step by step.

3. Loss Function:

The training is supervised by minimizing the difference (mean squared error) between the actual noise added in the forward process and the predicted noise.

4. Noise Schedule:

The variance β at each timestep defines how much noise to add. It can be a fixed schedule, e.g., linearly increasing over time.
"""

import einops
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import MSELoss


def forward_diffusion(x_0, t, noise_schedule):
    _ts = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_t = (
        noise_schedule["sqrtab"][_ts] * x_0
        + noise_schedule["sqrtmab"][_ts] * noise
    )
    return x_t, noise


def ddpm_schedule(beta1: float, beta2: float, T: int):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

def reverse_process(model, x_t, t):
    """
    x_t: noisy data at timestep t
    t: current timestep
    """
    return model(x_t, t)


def sample_by_denoising(model, x_T, β_schedule, T):
    """
    x_T: starting noisy data at timestep T
    Use the trained model to denoise and reconstruct data.
    """
    x_t = x_T
    for t in range(T, 0, -1):
        predicted_noise = reverse_process(model, x_t, t)
        x_t = (x_t - torch.sqrt(β_schedule[t]) * predicted_noise) / torch.sqrt(1 - β_schedule[t])
    return x_t  # Return reconstructed data


def random_timestep(T):
    return torch.randint(1, T, (1,))


def train_loop(model, dataloader, optimizer, β_schedule, T, num_train_steps, device):
    """
    model: neural network to predict noise at each time step
    optimizer: optimizer to update model parameters
    β_schedule: noise schedule
    T: total diffusion steps
    """
    model.train()
    loss_fn = MSELoss()
    for train_step in range(num_train_steps):
        batch = next(iter(dataloader))
        # x_0 = batch["image"]
        x_0, _ = batch
        x_0 = x_0.to(device)
        t = random_timestep(T).to(device)
        x_t, true_noise = forward_diffusion(x_0, t, β_schedule)
        print(f"x_0 max: {x_0.max()}, x_0 min: {x_0.min()}")
        print(f"x_t max: {x_t.max()}, x_t min: {x_t.min()}")
        predicted_noise = reverse_process(model, x_t, t / T)
        loss = loss_fn(predicted_noise, true_noise)
        print(f"[step {train_step}] Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        image_latent_dim = 64
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, image_latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, image_latent_dim),
            nn.ReLU(),
        )
        # (b, c, h, w) -> (b, c, h, w)
        self.decoder = nn.Sequential(
            nn.Conv2d(image_latent_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
        )

    def forward(self, x, t):
        # Convert t to float
        t = t.float()
        # make sure t is in the correct shape
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        elif len(t.shape) == 1:
            t = t.unsqueeze(0)
        else:
            raise ValueError(f"Invalid shape for t: {t.shape}")

        image_features = self.image_encoder(x)  # b c h w
        time_features = self.time_encoder(t)  # 1 d
        # tile time features to match image features
        time_features = einops.repeat(time_features, "1 d -> b d 1 1", b=x.shape[0],)
        print(f"image_features: {image_features.shape}, time_features: {time_features.shape}")

        # Concat or add features
        # features = torch.cat([image_features, time_features], dim=-1)
        features = image_features + time_features

        denoised_x = self.decoder(features)
        return denoised_x


def main():
    # Hyperparameters
    T = 1000
    num_train_steps = 1000
    device = "cuda:0"

    # load cifar10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # define the denoising model (convnet)
    denoising_model = DenoisingModel().to(device)
    
    # define the optimizer
    optimizer = optim.Adam(denoising_model.parameters(), lr=5e-5)

    # define the noise schedule
    noise_schedule = ddpm_schedule(1e-4, 0.02, T)
    for k, v in noise_schedule.items():
        noise_schedule[k] = v.to(device)

    # train the model
    train_loop(
        denoising_model, dataloader, optimizer, noise_schedule,
        T, num_train_steps, device=device)


if __name__ == "__main__":
    main()


