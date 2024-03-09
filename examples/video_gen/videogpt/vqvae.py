import math
import argparse
import os
import numpy as np
from typing import Tuple

import wandb
from einops import rearrange, pack
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import make_grid

from attention import MultiHeadAttention
from utils import shift_dim
from cvlization.torch.training_pipeline.image_gen.vae_resnet.vector_quantizers import (
    BaseVectorQuantizer,
)
import ae_variants


class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        self.network_variant = args.network_variant

        network_holder = getattr(ae_variants, self.network_variant)(
            embedding_dim=self.embedding_dim
        )
        self.encoder = network_holder["encode"]
        self.decoder = network_holder["decode"]
        self.vq = network_holder["vq"]

        # self.encoder = Encoder(args.n_hiddens, args.n_res_layers, args.downsample)
        # self.decoder = Decoder(args.n_hiddens, args.n_res_layers, args.downsample)
        # self.encoder = MiniEncoder(self.embedding_dim)
        # self.decoder = MiniDecoder(self.embedding_dim)
        # self.simple_conv = nn.Conv3d(3, 3, kernel_size=1, padding=0)

        # self.pre_vq_conv = SamePadConv3d(args.n_hiddens, args.embedding_dim, 1)
        # self.post_vq_conv = SamePadConv3d(args.embedding_dim, args.n_hiddens, 1)

        # self.codebook = Codebook(args.n_codes, args.embedding_dim)
        # self.codebook = Codebook2(args.n_codes, args.embedding_dim)
        # self.codebook = Codebook3(args.n_codes, args.embedding_dim)
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (
            self.args.sequence_length,
            self.args.resolution,
            self.args.resolution,
        )
        return tuple([s // d for s, d in zip(input_shape, self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        print("before vq", h.max(), h.min())
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings):

        h = F.embedding(encodings, self.codebook.embeddings)
        # print('after vq', h.max(), h.min())
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        # pre_vq_conv: (B, C, T, H, W) -> (B, C, T, H, W)
        # print(f"x: {x.shape}")
        z = self.encoder(x)
        # print(f"after encoding, z: {z.shape}")
        # z = self.pre_vq_conv(z)
        # print(f"z after pre_vq_conv: {z.shape}")

        vq_output = self.vq(z)
        # vq_output["z"] = z
        # z_recon = vq_output["embeddings"]
        if isinstance(vq_output, dict):
            z_recon = vq_output["z_recon"]
        else:
            z_recon = z
            vq_output = {
                "commitment_loss": 0,
                "perplexity": 0,
                "z": z,
                "avg_min_distance": 0,
            }

        # print(f"z_recon: {z_recon.shape}")
        # z_recon = self.post_vq_conv(z_recon)

        x_recon = self.decoder(z_recon)
        # x_recon = self.simple_conv(x)

        # print(f"x_recon: {x_recon.shape}")
        recon_loss_weight = 1.0  # TODO: make this an argument
        recon_loss = F.mse_loss(x_recon, x) * recon_loss_weight

        return recon_loss, x_recon, vq_output

    def on_train_start(self) -> None:
        print(self)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, x_recon, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss
        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True)
        self.log("train/commitment_loss", commitment_loss, on_step=True)
        self.log("train/loss", loss, on_step=True)
        self.log("train/perplexity", vq_output["perplexity"], on_step=True)
        self.log(
            "train/avg_min_vq_distance", vq_output["avg_min_distance"], on_step=True
        )
        self.log("train/z_mean", vq_output["z"].mean(), on_step=True)
        self.log("train/z_max", vq_output["z"].max(), on_step=True)
        self.log("train/z_min", vq_output["z"].min(), on_step=True)

        # log reconstructions (every 5 epochs, for one batch)
        if batch_idx == 2 and self.current_epoch % 5 == 0:
            if self.args.track:
                self.log_reconstructions(x, x_recon, training=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, x_recon, vq_output = self.forward(x)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/perplexity", vq_output["perplexity"], prog_bar=True)
        self.log("val/commitment_loss", vq_output["commitment_loss"], prog_bar=True)
        self.log("val/x_mean", x.mean())
        self.log("val/x_max", x.max())
        self.log("val/x_min", x.min())
        self.log("val/recon_mean", x_recon.mean())
        self.log("val/recon_max", x_recon.max())
        self.log("val/recon_min", x_recon.min())

        # log reconstructions (for one batch)
        if batch_idx == 2:
            if self.args.track:
                self.log_reconstructions(x, x_recon, training=False)

    @torch.no_grad()
    def preprocess_visualization(
        self,
        images: torch.Tensor,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """
        Preprocess images by de-normalizing back to range 0_1
        :param images: (B C H W) output of the autoencoder
        :param mean: 3 channels mean vector for de-normalization, default to imagenet values
        :param std: 3 channels std vector for de-normalization, default to imagenet values
        :return denormalized images in range 0__1
        """
        # images = denormalize(images, torch.tensor(mean), torch.tensor(std))
        images = torch.clip(
            images, 0, 1
        )  # if mean,std are correct, should have no effect
        return images

    @torch.no_grad()
    def log_reconstructions(
        self, ground_truths, reconstructions, training: bool = True
    ):
        """
        Log the reconstructions to wandb.

        :param ground_truths: (B, T, C, H, W) ground truth videos
        :param reconstructions: (B, T, C, H, W) reconstructed videos
        :param training: whether the images are from the training set.
        """
        import wandb

        # make sure the shape is corect
        if ground_truths.shape[2] != 3:
            ground_truths = rearrange(ground_truths, "b t c h w -> b c t h w")
        if reconstructions.shape[2] != 3:
            reconstructions = rearrange(reconstructions, "b t c h w -> b c t h w")

        # make sure the device is cpu
        if ground_truths.device.type != "cpu":
            ground_truths = ground_truths.cpu()
        if reconstructions.device.type != "cpu":
            reconstructions = reconstructions.cpu()

        # make sure the pixel values are in the range [0, 1]
        residual = reconstructions - ground_truths
        ground_truths = (ground_truths - ground_truths.min()) / (
            ground_truths.max() - ground_truths.min() + 1e-6
        )
        reconstructions = (reconstructions - reconstructions.min()) / (
            reconstructions.max() - reconstructions.min() + 1e-6
        )
        residual = (residual - residual.min()) / (
            residual.max() - residual.min() + 1e-6
        )
        ground_truths = (ground_truths * 255).to(torch.uint8)
        reconstructions = (reconstructions * 255).to(torch.uint8)
        residual = (residual * 255).to(torch.uint8)

        # b = min(ground_truths.shape[0], 8)
        b = min(ground_truths.shape[0], 2)
        # b = min(ground_truths.shape[0], 1)
        panel_name = "train" if training else "validation"

        # side by side video
        display = torch.cat(
            [ground_truths[:b], reconstructions[:b], residual[:b]], dim=3
        )

        display = wandb.Video(data_or_path=display, fps=4, format="mp4")
        self.logger.experiment.log({f"{panel_name}/reconstructions": display})

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # reconstruct and visualize

        # log the model
        # model_artifact = wandb.Artifact(
        #     "vqvae",
        #     type="model",
        #     description="VQVAE",
        #     metadata={
        #         "epoch": self.current_epoch,
        #         "val/recon_loss": self.trainer.callback_metrics["val/recon_loss"],
        #         "val/perplexity": self.trainer.callback_metrics["val/perplexity"],
        #         "val/commitment_loss": self.trainer.callback_metrics[
        #             "val/commitment_loss"
        #         ],
        #     },
        # )
        # current_epoch = self.current_epoch
        # if current_epoch >= 10:
        #     os.makedirs("logs", exist_ok=True)
        #     torch.save(self.state_dict(), "logs/vqvae.pth")
        #     model_artifact.add_file("logs/vqvae.pth")
        #     self.logger.experiment.log_artifact(model_artifact)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_codes", type=int, default=2048)
        # parser.add_argument("--n_hiddens", type=int, default=240)
        parser.add_argument("--n_hiddens", type=int, default=256)
        parser.add_argument("--n_res_layers", type=int, default=4)
        parser.add_argument("--downsample", nargs="+", type=int, default=(4, 4, 4))
        return parser


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=n_head,
            n_layer=1,
            causal=False,
            attn_type="axial",
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2),
        )

    def forward(self, x):
        return x + self.block(x)


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


class MiniEncoder(nn.Module):
    """
    A minimal spatial-temporal encoder for testing purposes.
    """

    def __init__(self, embedding_dim: int = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (B, C, T, H, W) -> (B, C, T, H/2, W/2) -> (B, C, T, H/4, W/4) -> (B, C, T/2, H/8, W/8)
        self.conv0 = nn.Conv3d(
            3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.conv_1by1 = nn.Conv3d(
            3, embedding_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv1 = nn.Conv3d(
            3, 16, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            16, 32, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.conv3 = nn.Conv3d(
            32, embedding_dim, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.conv_1by1(x)
        return self.conv0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x


class MiniDecoder(nn.Module):
    """
    A minimal spatial-temporal decoder for testing purposes.
    """

    def __init__(self, embedding_dim: int = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

        self.relu = nn.ReLU()
        self.u1 = nn.Upsample(scale_factor=(1, 1, 1), mode="trilinear")
        self.c0 = nn.Conv3d(
            embedding_dim, 3, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.c1 = nn.Conv3d(
            embedding_dim, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.u2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        self.c2 = nn.Conv3d(32, 16, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.u3 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        self.c3 = nn.Conv3d(16, 3, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.d0 = nn.ConvTranspose3d(
            embedding_dim,
            16,
            kernel_size=[1, 4, 4],
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.cd0 = nn.Conv3d(
            16, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1by1 = nn.Conv3d(
            embedding_dim, 3, kernel_size=1, stride=(1, 1, 1), padding=(0, 0, 0)
        )

    def forward(self, x):
        # return self.conv_1by1(x)
        return self.cd0(self.d0(x))
        return self.c0(self.u2(x))
        x = self.u1(x)
        x = self.c1(x)
        x = self.relu(x)
        x = self.u2(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.u3(x)
        x = self.c3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


class Codebook3(BaseVectorQuantizer):
    # adapted from cvlization/torch/training_pipeline/image_gen/vae_resnet/vector_quantizers.py
    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25
    ):
        """
        Original VectorQuantizer with straight through gradient estimator (loss is optimized on inputs and codebook)
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param commitment_cost: scaling factor for e_loss
        """
        print(
            f"***** Creating a quantizer with {num_embeddings} embeddings and {embedding_dim} dimensions"
        )
        super().__init__(num_embeddings, embedding_dim)

        self.commitment_cost = commitment_cost

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
        # distances is a matrix (B*H*W, codebook_size)
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
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  # ?

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
        embedings_st = quantized
        commitment_loss = q_loss + e_loss  # ?
        avg_probs = torch.mean(encoding_indices.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_distances = torch.min(distances, dim=1).values
        avg_min_distance = torch.mean(min_distances)
        return dict(
            # embeddings=embedings_st,
            embeddings=x,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
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


class Codebook2(nn.Module):
    # This is adapted from CVlization/examples/image_gen/vqgan/codebook.py
    def __init__(self, n_codes: int, embedding_dim: int, beta: float = 0.25):
        super(Codebook2, self).__init__()
        self.num_codebook_vectors = n_codes
        self.latent_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors
        )

    def forward(self, z):
        # z is of shape (b, c, t, h, w), c is the embedding dim
        z = z.permute(0, 2, 3, 4, 1).contiguous()  # (b, t, h, w, c)
        # print(f"z: {z.shape}")
        z_flattened = z.view(-1, self.latent_dim)
        # print(f"b * t * h * w: {z_flattened.shape[0]}")

        # this is the distance between the z and the embeddings
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * (torch.matmul(z_flattened, self.embedding.weight.t()))
        )  # (b*t*h*w, c)
        assert d.shape == (z_flattened.shape[0], self.num_codebook_vectors)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        # print(f"z_q: {z_q.shape}")
        # permute back
        z_q = z_q.permute(0, 4, 1, 2, 3)  # (b, c, t, h, w)

        # return z_q, min_encoding_indices, loss
        embedings_st = z_q
        # print(f"z_q: {z_q.shape}, min_encoding_indices: {min_encoding_indices.shape}")
        encoding_indices = min_encoding_indices.view(z_q.shape[0], *z_q.shape[2:])
        commitment_loss = loss
        perplexity = torch.exp(
            -torch.mean(
                torch.sum(
                    F.one_hot(min_encoding_indices, self.num_codebook_vectors).float()
                    * F.log_softmax(d, dim=1),
                    dim=1,
                )
            )
        )
        return dict(
            embeddings=embedings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )
