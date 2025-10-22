"""

Simple DDPM noise schedule, a simple convnet as the denoising model.

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

from pathlib import Path
import einops
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchvision.utils import save_image, make_grid

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


def sample_by_denoising(model, x_T, noise_schedule, T):
    """
    x_T: starting noisy data at timestep T
    Use the trained model to denoise and reconstruct data.
    """
    x_t = x_T
    for t in range(T, 0, -1):
        z = torch.randn_like(x_t) if t > 1 else 0
        t_tensor = torch.tensor([t], dtype=torch.float32).to(x_t.device)
        predicted_noise = reverse_process(model, x_t, t_tensor / T)
        # x_i = (
        #     self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
        #     + self.sqrt_beta_t[i] * z
        # )
        oneover_sqrta = noise_schedule["oneover_sqrta"][t]
        mab_over_sqrtmab = noise_schedule["mab_over_sqrtmab"][t]
        x_t = (
            oneover_sqrta * (x_t - predicted_noise * mab_over_sqrtmab)
            + noise_schedule["sqrt_beta_t"][t] * z
        )
    return x_t  # Return reconstructed data


def random_timestep(T):
    return torch.randint(1, T, (1,))


def train_loop(model, dataloader, optimizer, noise_schedule, T, num_train_steps, device, log_every=10, sample_every=100):
    """
    model: neural network to predict noise at each time step
    optimizer: optimizer to update model parameters
    β_schedule: noise schedule
    T: total diffusion steps
    device: device to run the computations on
    log_every: number of steps between logging loss
    sample_every: number of steps between generating sample images
    """
    model.train()
    loss_fn = MSELoss()
    for train_step in range(num_train_steps):
        batch = next(iter(dataloader))
        x_0, _ = batch
        x_0 = x_0.to(device)
        t = random_timestep(T).to(device)
        x_t, true_noise = forward_diffusion(x_0, t, noise_schedule)
        
        predicted_noise = model(x_t, t / T)
        loss = loss_fn(predicted_noise, true_noise)
        
        if train_step % log_every == 0:
            print(f"[step {train_step}] Loss: {loss.item():.4f}")
        
        if train_step % sample_every == 0:
            # Generate and save a sample image
            num_samples = 8
            x_T = torch.randn(num_samples, 3, 32, 32).to(device)
            model.eval()
            sampled_images = sample_by_denoising(model, x_T, noise_schedule, T)
            model.train()
            to_display = torch.cat([sampled_images, x_0[:num_samples]], dim=0)
            grid = make_grid(to_display, normalize=True, value_range=(-1, 1), nrow=4)
            img_output_dir = Path("data/image_gen/nano_diffusion/train1")
            img_output_dir.mkdir(parents=True, exist_ok=True)
            save_image(grid, img_output_dir / f"ddpm_sample_cifar_{train_step}.png")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        image_latent_dim = 64
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, image_latent_dim, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(image_latent_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
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

        # Concat or add features
        features = image_features + time_features

        denoised_x = self.decoder(features)
        return denoised_x


def main():
    # Hyperparameters
    T = 1000
    num_train_steps = 10000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_every = 20
    sample_every = 500

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
        T, num_train_steps, device, log_every, sample_every
    )


if __name__ == "__main__":
    main()


