import numpy as np
import torch
from torch import nn
import math


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimestepEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # Create a tensor of positional frequencies
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        # Calculate the sinusoidal embeddings
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad if odd dimension
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class SampleInitializer(nn.Module):
    def __init__(self, latent_dim, img_shape, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.hidden_dim = hidden_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(img_shape[0], hidden_dim, kernel_size=5, stride=2, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1
            ),
        )

        self.time_encoder = TimestepEmbedding(hidden_dim)

        self.image_decoder = nn.Sequential(
            [
                nn.ConvTranspose2d(
                    hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                    hidden_dim, img_shape[0], kernel_size=5, stride=2, padding=4
                ),
                nn.Tanh(),
            ]
        )

    def encode(self, x, timesteps):
        """
        x: input image
        timesteps: diffusion timesteps (noise levels)
        """
        # convnet
        x = self.image_encoder(x)
        timesteps = self.time_encoder(timesteps)
        # x = x + timesteps
        z = torch.cat([x, timesteps], dim=1)
        return z

    def decode(self, z, timesteps=None):
        """
        z: latent vector
        timesteps: diffusion timesteps (noise levels)
        """
        # convnet
        x = self.image_decoder(z)
        return x

    def forward(self, x, t):
        z = self.encode(x, t)
        y = self.decode(z, t)
        return y


class DecoderGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 8, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
