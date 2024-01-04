import random
import torch
import torch.nn as nn
from torch import optim
import math
import torchvision
import lightning.pytorch as pl
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from .sampler import Sampler
from .generator import TimestepEmbedding, SampleInitializer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):
    def __init__(
        self,
        hidden_features=32,
        time_embedding_dim=32,
        out_dim=1,
        n_channels=1,
        **kwargs,
    ):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(
                n_channels, c_hid1, kernel_size=5, stride=2, padding=4
            ),  # [16x16] - Larger padding to get 32x32 image
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  #  [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=1, padding=1),  # [2x2]
            Swish(),
        )
        self.embed_time = TimestepEmbedding(time_embedding_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, timesteps):
        assert str(x.device).startswith(
            "cuda"
        ), f"inputs to the model must be on the GPU. Got {x.device}"
        image_embedding = self.cnn_layers(x)
        time_embedding = self.embed_time(timesteps)
        concat_embedding = torch.concat([image_embedding, time_embedding], dim=1)
        x = self.pool(concat_embedding)
        x = x.squeeze(dim=-1)
        return x


class DeepEnergyModel(pl.LightningModule):
    def __init__(
        self,
        img_shape,
        batch_size,
        diffusion_num_steps=100,
        alpha=0.1,
        lr=1e-4,
        beta1=0.0,
        **CNN_args,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = CNNModel(n_channels=img_shape[0], **CNN_args)
        # self.cnn.to("cuda")
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_num_steps,
            beta_schedule="linear",
        )

    def forward(self, x, t):
        """
        Args:
            x: input image
            t: diffusion timestep, or noise level
        Returns:
            energy (scalar)
        """
        assert str(x.device).startswith(
            "cuda"
        ), f"inputs to the model must be on the GPU. Got {x.device}"
        z = self.cnn(x, t)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999)
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.97
        )  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def _generate_noisy_images_from_real_images(self, real_imgs):
        clean_images = real_imgs
        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images, timesteps

    def _run_mcmc_langevin_to_generate_fake_images(
        self, noisy_imgs, timesteps, num_mcmc_steps=20
    ):
        fake_images = self.sampler.sample_new_exmps(
            noisy_imgs, timesteps, steps=num_mcmc_steps, step_size=10
        )
        return fake_images

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        # small_noise = torch.randn_like(real_imgs) * 0.005
        # real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # This is the diffusion step to add noise to data.
        noisy_imgs, timesteps = self._generate_noisy_images_from_real_images(
            real_imgs, timesteps
        )

        # This is the MCMC step to generate fake images.
        fake_imgs = self._run_mcmc_langevin_to_generate_fake_images(
            noisy_imgs, timesteps
        )

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log("loss", loss)
        self.log("loss_regularization", reg_loss)
        self.log("loss_contrastive_divergence", cdiv_loss)
        self.log("metrics_avg_real", real_out.mean())
        self.log("metrics_avg_fake", fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("val_contrastive_divergence", cdiv)
        self.log("val_fake_out", fake_out.mean())
        self.log("val_real_out", real_out.mean())


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size  # Number of images to generate
        self.vis_steps = vis_steps  # Number of steps within generation to visualize
        self.num_steps = num_steps  # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size - 1 :: step_size, i]
                grid = torchvision.utils.make_grid(
                    imgs_to_plot,
                    nrow=imgs_to_plot.shape[0],
                    normalize=True,
                    range=(-1, 1),
                )
                trainer.logger.experiment.add_image(
                    f"generation_{i}", grid, global_step=trainer.current_epoch
                )

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(
            pl_module.device
        )
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(
            pl_module.cnn,
            start_imgs,
            steps=self.num_steps,
            step_size=10,
            return_img_per_step=True,
        )
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step


class SamplerCallback(pl.Callback):
    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs  # Number of images to plot
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        # raise NotImplementedError("This callback is not yet implemented")
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(
                random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0
            )
            grid = torchvision.utils.make_grid(
                exmp_imgs, nrow=4, normalize=True, range=(-1, 1)
            )
            trainer.logger.experiment.add_image(
                "sampler", grid, global_step=trainer.current_epoch
            )
            print(f"Saved sampler images to {trainer.logger}")
        else:
            print(f"Skipped saving sampler images to {trainer.logger}")


class OutlierCallback(pl.Callback):
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_validation_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand(
                (self.batch_size,) + pl_module.hparams["img_shape"]
            ).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar(
            "rand_out", rand_out, global_step=trainer.current_epoch
        )
