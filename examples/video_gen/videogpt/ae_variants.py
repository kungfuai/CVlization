import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange, pack
from cvlization.torch.training_pipeline.image_gen.vae_resnet.vector_quantizers import (
    BaseVectorQuantizer,
)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class VariationalEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layers_before_last = torch.nn.Sequential(*encoder_layers[:-1])

        self.mu_layer = encoder_layers[-1]
        # create a layer for the logvar, with the same shape as the mu layer
        self.sigma_layer = nn.Conv3d(
            self.mu_layer.in_channels,
            self.mu_layer.out_channels,
            kernel_size=self.mu_layer.kernel_size,
            stride=self.mu_layer.stride,
            padding=self.mu_layer.padding,
        )

    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.encoder_layers_before_last(x)
        mu = self.mu_layer(x)
        logvar = self.sigma_layer(x)
        z = self.sample(mu, logvar)
        return dict(mu=mu, logvar=logvar, z=z)

def encode111111_decode111111(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
    )
    decode = torch.nn.Conv3d(
        embedding_dim, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode133111_decode111111(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
    )
    decode = torch.nn.Conv3d(
        embedding_dim, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode133122_decode144122(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    d0 = torch.nn.ConvTranspose3d(
        embedding_dim,
        16,
        kernel_size=[1, 4, 4],
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )
    cd0 = torch.nn.Conv3d(
        16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
    )
    decode = torch.nn.Sequential(d0, cd0)
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode133122_decode144122_tanh(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    d0 = torch.nn.ConvTranspose3d(
        embedding_dim,
        16,
        kernel_size=[1, 4, 4],
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )
    cd0 = torch.nn.Conv3d(
        16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(d0, cd0, tanh)
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode133122_decode144122_tanh_vq(
    embedding_dim: int = 8,
    num_embeddings: int = 512,
    low_utilization_cost: float = 0,
    commitment_cost: float = 0.25,
    **kwargs,
):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
    )
    d0 = torch.nn.ConvTranspose3d(
        embedding_dim,
        16,
        kernel_size=[1, 4, 4],
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )
    cd0 = torch.nn.Conv3d(
        16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(d0, cd0, tanh)
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        low_utilization_cost=low_utilization_cost,
        commitment_cost=commitment_cost,
    )
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode333222_decode444222_tanh(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Conv3d(
        3, embedding_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
    )
    d0 = torch.nn.ConvTranspose3d(
        embedding_dim,
        16,
        kernel_size=[4, 4, 4],
        stride=(2, 2, 2),
        padding=(1, 1, 1),
    )
    cd0 = torch.nn.Conv3d(
        16, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(d0, cd0, tanh)
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode_decode_spatial4x(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            3,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def s4t4(embedding_dim: int = 8):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            3,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode_decode_spatial4x_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def vae_spatial4x_a(embedding_dim: int = 8, **kargs):
    encoder_layers = [
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    ]
    encode = VariationalEncoder(encoder_layers)
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def encode_decode_spatial4x_a_vq(
    embedding_dim: int = 8,
    num_embeddings: int = 512,
    low_utilization_cost: float = 0,
    commitment_cost: float = 0.25,
    **kwargs,
):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        low_utilization_cost=low_utilization_cost,
        commitment_cost=commitment_cost,
    )
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def vae_spatial4x_a_vq(embedding_dim: int = 8, num_embeddings: int = 512, low_utilization_cost: float = 0, commitment_cost: float = 0.25, **kargs):
    encoder_layers = [
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    ]
    encode = VariationalEncoder(encoder_layers)
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        low_utilization_cost=low_utilization_cost,
        commitment_cost=commitment_cost,
    )
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def s4t4_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            16, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def s4t4_b(embedding_dim: int = 8, **kwargs):
    latent_dims = [32]
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3,
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            3,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def s4t4_c(embedding_dim: int = 8, **kwargs):
    """
    Add residual connections to s4t4_b
    """
    latent_dims = [32]
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3,
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        ResidualLayer(embedding_dim, embedding_dim, latent_dims[0]),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        ResidualLayer(embedding_dim, embedding_dim, latent_dims[0]),
        torch.nn.ConvTranspose3d(
            embedding_dim,
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            3,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def s4t4_b_vq(
    embedding_dim: int = 8,
    num_embeddings: int = 512,
    low_utilization_cost: float = 0,
    commitment_cost: float = 0.25,
    **kwargs,
):
    latent_dims = [32]
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3,
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            3,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        tanh,
    )
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        low_utilization_cost=low_utilization_cost,
        commitment_cost=commitment_cost,
    )
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }

def vae_s4t4_b_vq(num_embeddings: int = 512, low_utilization_cost: float = 0, commitment_cost: float = 0.25, **kargs):
    latent_dims = [32]
    encoder_layers = [
        torch.nn.Conv3d(
            3,
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    ]
    encode = VariationalEncoder(encoder_layers)
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            latent_dims[0],
            latent_dims[0],
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.LeakyReLU(),
        torch.nn.Conv3d(
            latent_dims[0],
            3,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        tanh,
    )
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=latent_dims[0],
        low_utilization_cost=low_utilization_cost,
        commitment_cost=commitment_cost,
    )
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": latent_dims[0],
    }

def encode_decode_spatial8x_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            32,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            32,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            32,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            32,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def s8t8_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            32,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            32,
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            32,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            32,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def encode_decode_spatial16x_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            32,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            32,
            64,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            64,
            embedding_dim,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            64,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            64,
            32,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            32,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


def s16t16_a(embedding_dim: int = 8, **kwargs):
    encode = torch.nn.Sequential(
        torch.nn.Conv3d(
            3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            16,
            32,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            32,
            64,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.ReLU(),
        torch.nn.Conv3d(
            64,
            embedding_dim,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
    )
    tanh = torch.nn.Tanh()
    decode = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(
            embedding_dim,
            64,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            64,
            32,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            32,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        torch.nn.LeakyReLU(),
        torch.nn.ConvTranspose3d(
            16,
            16,
            kernel_size=[4, 4, 4],
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        ),
        torch.nn.Conv3d(
            16, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        ),
        tanh,
    )
    vq = torch.nn.Identity()
    return {
        "encode": encode,
        "decode": decode,
        "vq": vq,
        "embedding_dim": embedding_dim,
    }


class VectorQuantizer(BaseVectorQuantizer):
    # adapted from cvlization/torch/training_pipeline/image_gen/vae_resnet/vector_quantizers.py
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        low_utilization_cost: float = 0.0,
    ):
        """
        Original VectorQuantizer with straight through gradient estimator (loss is optimized on inputs and codebook)
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param commitment_cost: scaling factor for e_loss
        :param low_utilization_cost: scaling factor for codebook_utilization_loss
        """
        print(
            f"***** Creating a quantizer with {num_embeddings} embeddings and {embedding_dim} dimensions"
        )
        super().__init__(num_embeddings, embedding_dim)

        self.commitment_cost = commitment_cost
        self.low_utilization_cost = low_utilization_cost

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.IntTensor, float]:
        """
        :param x: spatial temporal tensors (output of the Encoder - B,C,T,H,W).
        :return quantized_x (B,C,T,H,W), encoding_indices (B,T,H,W), loss (float)
        """
        b, c, t, h, w = x.shape
        device = x.device

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c t h w -> (b t h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*T*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Loss functions
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)
        q_loss = F.mse_loss(quantized, flat_x.detach())

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        quantized = rearrange(
            quantized, "(b t h w) c -> b c t h w", b=b, h=h, w=w, t=t, c=c
        )
        encoding_indices = rearrange(
            encoding_indices, "(b t h w)-> b (t h w)", b=b, h=h, w=w, t=t
        ).detach()

        # return quantized, encoding_indices, q_loss + e_loss
        # embedings_st = quantized
        commitment_loss = q_loss + e_loss  # ?
        if self.training and self.low_utilization_cost > 0:
            # Now, in reverse, project the codebook onto the input space
            # and calculate the loss
            # print(f"******** applying low_utilization_cost {self.low_utilization_cost}")
            codebook_indices_onto_input = torch.argmin(
                distances, dim=0
            )  # shape is (codebook_size)
            codebook_encodings_onto_input = torch.zeros(
                self.num_embeddings, b * t * h * w, device=device
            )
            # TODO: fix this!
            # scatter_:
            # For 3D tensor,
            # - If dim == 0, dst[ids[i][j][k]][j][k] = src[i][j][k]
            # - If dim == 1, dst[i][ids[i][j][k]][k] = src[i][j][k]
            # - If dim == 2, dst[i][j][ids[i][j][k]] = src[i][j][k]
            # For 2D tensor,
            # - If dim == 0, dst[ids[i][j]][j] = src[i][j]
            # - If dim == 1, dst[i][ids[i][j]] = src[i][j]
            # We want codebook_encodings_onto_input to be a 2D tensor.
            # codebook_encodings_onto_input[i][j] = 1 if codebook_indices_onto_input[i] == j else 0
            # In other words,
            # codebook_encodings_onto_input[i][indices[j]] = 1
            codebook_encodings_onto_input.scatter_(
                dim=1,
                index=codebook_indices_onto_input.unsqueeze(1),
                src=torch.ones(
                    (self.num_embeddings, b * t * h * w), device=device
                ),
            )
            # flat_x shape is (B*T*H*W, embedding_dim)
            # distances shape is (B*T*H*W, num_embeddings)
            # print(
            #     f"flat_x shape is {flat_x.shape}, codebook_encodings_onto_input shape is {codebook_encodings_onto_input.shape}"
            # )
            codebook_quantized = torch.matmul(
                codebook_encodings_onto_input, flat_x
            )  # shape is (num_embeddings, embedding_dim)
            # codebook_quantized = (
            #     self.codebook.weight
            #     + (codebook_quantized - self.codebook.weight).detach()
            # )
            codebook_utilization_loss = F.mse_loss(
                codebook_quantized.detach(), self.codebook.weight
            )
            commitment_loss += codebook_utilization_loss * self.low_utilization_cost
        min_distances = torch.min(distances, dim=1).values
        avg_min_distance = torch.mean(min_distances)
        return dict(
            z_recon=quantized,
            z=x,
            embeddings=x,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            avg_min_distance=avg_min_distance,
        )

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,C,T,H,W).
        :return flat codebook indices (B, T * H * W)
        """
        b, c, t, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c t h w -> (b t h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(
            encoding_indices, "(b t h w) -> b (t h w)", b=b, h=h, w=w, t=t
        )

        return encoding_indices

    @torch.no_grad()
    def vec_to_codes_2d(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        b, c, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, "b c h w -> (b h w) c")

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.weight.t())
        )

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(
            encoding_indices, "(b h w) -> b (h w)", b=b, h=h, w=w
        )

        return encoding_indices


class VectorQuantizer2(torch.nn.Module):
    """
    From https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
