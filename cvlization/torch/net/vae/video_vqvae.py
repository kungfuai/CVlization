import argparse
import os

from einops import rearrange
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvlization.torch.net.vae import video_vae_components as ae_variants


def load_model_from_wandb(
    model_full_name: str = "zzsi_kungfu/videogpt/model-tjzu02pg:v17",
) -> dict:
    import wandb

    api = wandb.Api()
    # skip if the file already exists
    artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
    if os.path.exists(artifact_dir):
        print(f"Model already exists at {artifact_dir}")
    else:
        artifact_dir = api.artifact(model_full_name).download()
    # The file is model.ckpt.
    state_dict = torch.load(artifact_dir + "/model.ckpt")
    # print(list(state_dict.keys()))
    hyper_parameters = state_dict["hyper_parameters"]
    args = hyper_parameters["args"]
    from cvlization.torch.net.vae.video_vqvae import VQVAE

    # args = Namespace(**hyper_parameters)
    # print(args)
    model = VQVAE.load_from_checkpoint(artifact_dir + "/model.ckpt")
    # model = VQVAE(args=args)
    # model.load_state_dict(state_dict["state_dict"])
    return model


class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        self.network_variant = args.network_variant

        network = getattr(ae_variants, self.network_variant)(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.n_codes,
            low_utilization_cost=args.low_utilization_cost,
            commitment_cost=args.commitment_cost,
        )
        self.encoder = network["encode"]
        self.decoder = network["decode"]
        self.vq = network["vq"]
        self.train_epoch_usage_count = None
        self.val_epoch_usage_count = None
        self.kl_loss_weight = args.kl_loss_weight

        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (
            self.args.sequence_length,
            self.args.resolution,
            self.args.resolution,
        )
        return tuple([s // d for s, d in zip(input_shape, self.args.downsample)])

    @classmethod
    def from_pretrained(cls, model_full_name: str):
        """
        Load a pretrained model from wandb.

        :param model_full_name: the full name of the model on wandb. e.g. "zzsi_kungfu/videogpt/model-tjzu02pg:v17"
        """
        import wandb

        api = wandb.Api()
        # skip if the file already exists
        artifact_dir = f"artifacts/{model_full_name.split('/')[-1]}"
        if os.path.exists(artifact_dir):
            print(f"Model already exists at {artifact_dir}")
        else:
            artifact_dir = api.artifact(model_full_name).download()
        # The file is model.ckpt.
        state_dict = torch.load(artifact_dir + "/model.ckpt")
        hyper_parameters = state_dict["hyper_parameters"]
        args = hyper_parameters["args"]
        model = cls(args=args)
        model.load_state_dict(state_dict["state_dict"])
        return model

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # pre_vq_conv: (B, C, T, H, W) -> (B, C, T, H, W)
        # print(f"x: {x.shape}")
        encoded = self.encoder(x)
        if isinstance(encoded, dict):
            z = encoded["z"]
            mu = encoded["mu"]
            logvar = encoded["logvar"]
            kl_loss = (
                -0.5
                * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                * self.kl_loss_weight
            )
        elif isinstance(encoded, torch.Tensor):
            z = encoded
            kl_loss = 0
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
                "z": z,
                "avg_min_distance": 0,
            }

        # print(f"z_recon: {z_recon.shape}")
        # z_recon = self.post_vq_conv(z_recon)

        x_recon = self.decoder(z_recon)
        # x_recon = self.simple_conv(x)

        # print(f"x_recon: {x_recon.shape}")
        recon_loss_weight = 1.0  # TODO: make this an argument
        # print(f"x: {x.shape}, x_recon: {x_recon.shape}")
        recon_loss = F.mse_loss(x_recon, x) * recon_loss_weight

        return recon_loss, kl_loss, x_recon, vq_output

    def on_train_start(self) -> None:
        print(self)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, kl_loss, x_recon, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss + kl_loss
        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True)
        self.log("train/commitment_loss", commitment_loss, on_step=True)
        self.log("train/kl_loss", kl_loss, on_step=True)
        self.log("train/loss", loss, on_step=True)
        self.log(
            "train/avg_min_vq_distance", vq_output["avg_min_distance"], on_step=True
        )
        self.log("train/z_mean", vq_output["z"].mean(), on_step=True)
        self.log("train/z_max", vq_output["z"].max(), on_step=True)
        self.log("train/z_min", vq_output["z"].min(), on_step=True)
        self.log(
            "train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True
        )

        # log reconstructions (every 5 epochs, for one batch)
        if batch_idx == 2 and self.current_epoch % 5 == 0:
            if self.args.track:
                self.log_reconstructions(x, x_recon, training=True)

        if hasattr(self.vq, "get_codebook_usage"):
            # update codebook usage
            # batch index count (use non-deterministic for this operation)
            # torch.use_deterministic_algorithms(False)
            used_indices = vq_output["encodings"]
            used_indices = torch.bincount(
                used_indices.view(-1), minlength=self.args.n_codes
            )
            # torch.use_deterministic_algorithms(True)
            self.train_epoch_usage_count = (
                used_indices if self.train_epoch_usage_count is None else +used_indices
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, kl_loss, x_recon, vq_output = self.forward(x)
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/commitment_loss", vq_output["commitment_loss"], prog_bar=True)
        self.log("val/kl_loss", kl_loss)
        self.log("val/x_mean", x.mean())
        self.log("val/x_max", x.max())
        self.log("val/x_min", x.min())
        self.log("val/recon_mean", x_recon.mean())
        self.log("val/recon_max", x_recon.max())
        self.log("val/recon_min", x_recon.min())

        # log reconstructions (for one batch)
        if batch_idx == 0:
            if self.args.track:
                self.log_reconstructions(x, x_recon, training=False)

        if hasattr(self.vq, "get_codebook_usage"):
            # update codebook usage
            used_indices = vq_output["encodings"]
            used_indices = torch.bincount(
                used_indices.view(-1), minlength=self.args.n_codes
            )
            self.val_epoch_usage_count = (
                used_indices if self.val_epoch_usage_count is None else +used_indices
            )

    def on_train_epoch_end(self):
        if (
            hasattr(self.vq, "get_codebook_usage")
            and self.args.reinit_every_n_epochs is not None
            and self.current_epoch % self.args.reinit_every_n_epochs == 0
            and self.current_epoch > 0
        ):
            self.vq.reinit_unused_codes(
                self.vq.get_codebook_usage(self.train_epoch_usage_count)[0]
            )

        self.train_epoch_usage_count = None

    def on_validation_end(self) -> None:
        if hasattr(self.vq, "get_codebook_usage"):
            _, perplexity, cb_usage = self.vq.get_codebook_usage(
                self.val_epoch_usage_count
            )
            if hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log({"val/perplexity": perplexity})
                self.logger.experiment.log({"val/codebook_usage": cb_usage})
            elif hasattr(self.logger, "log"):
                self.logger.log({"val/perplexity": perplexity})
                self.logger.log({"val/codebook_usage": cb_usage})
            self.val_epoch_usage_count = None
        return super().on_validation_end()

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
        # b = min(ground_truths.shape[0], 2)
        b = min(ground_truths.shape[0], 1)
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.999)
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/recon_loss",
        }

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
