import torch


def encode111111_decode111111(embedding_dim: int = 8):
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


def encode133111_decode111111(embedding_dim: int = 8):
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


def encode133122_decode144122(embedding_dim: int = 8):
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


def encode133122_decode144122_tanh(embedding_dim: int = 8):
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


def encode_decode_spatial4x(embedding_dim: int = 8):
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


def encode_decode_spatial4x_a(embedding_dim: int = 8):
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


def encode_decode_spatial8x_a(embedding_dim: int = 8):
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


def encode_decode_spatial16x_a(embedding_dim: int = 8):
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
