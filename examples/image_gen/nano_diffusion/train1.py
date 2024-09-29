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

from pathlib import Path
import einops
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torchvision.utils import make_grid, save_image


def forward_diffusion(x_0, t, β_schedule):
    """
    x_0: initial data (e.g., image)
    t: timestep
    β_schedule: noise schedule
    
    Returns the noisy data at timestep t, and the noise added.
    """
    β_t = β_schedule[t]
    noise = torch.randn_like(x_0)
    return torch.sqrt(1 - β_t) * x_0 + torch.sqrt(β_t) * noise, noise


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
    for t in range(T - 1, 0, -1):
        t_tensor = torch.tensor([t]).float().to(x_T.device)
        predicted_noise = reverse_process(model, x_t, t_tensor)
        x_t = (x_t - torch.sqrt(β_schedule[t]) * predicted_noise) / torch.sqrt(1 - β_schedule[t])
    return x_t  # Return reconstructed data


def random_timestep(T):
    return torch.randint(1, T, (1,))


def train_loop(model, dataloader, optimizer, β_schedule, T, num_train_steps, device, log_every=20, sample_every=100):
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
        x_t, true_noise = forward_diffusion(x_0, t, β_schedule)
        
        predicted_noise = reverse_process(model, x_t, t / T)
        loss = loss_fn(predicted_noise, true_noise)
        
        if train_step % log_every == 0:
            print(f"[step {train_step}] Loss: {loss.item():.4f}")
            # print(f"x_0 max: {x_0.max()}, x_0 min: {x_0.min()}")
            # print(f"x_t max: {x_t.max()}, x_t min: {x_t.min()}")
        
        if train_step % sample_every == 0:
            # Generate and save a sample image
            num_samples = 2
            x_T = torch.randn(num_samples, 3, 32, 32).to(device)
            sampled_images = sample_by_denoising(model, x_T, β_schedule, T)
            to_display = torch.cat([sampled_images, x_0[:num_samples]], dim=0)
            grid = make_grid(to_display, normalize=True, value_range=(-1, 1), nrow=2)
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
        t = torch.tensor(t).float()
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
        # print(f"image_features: {image_features.shape}, time_features: {time_features.shape}")

        # Concat or add features
        # features = torch.cat([image_features, time_features], dim=-1)
        features = image_features + time_features

        denoised_x = self.decoder(features)
        return denoised_x


def main():
    # Hyperparameters
    T = 1000
    num_train_steps = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_every = 20
    sample_every = 100

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
    β_schedule = torch.linspace(0.0001, 0.02, T).to(device)

    # train the model
    train_loop(denoising_model, dataloader, optimizer, β_schedule, T, num_train_steps, device, log_every, sample_every)


if __name__ == "__main__":
    main()
