"""
Adapted from https://github.com/SerezD/vqvae-vqgan-pytorch-lightning
"""

from dataclasses import dataclass
import math
import os
from typing import List, Tuple, Any
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torchvision.utils import make_grid
from einops import rearrange, pack
from scheduling_utils.schedulers_cpp import (
    LinearCosineScheduler,
    LinearScheduler,
    CosineScheduler,
)
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.transforms import ConvertImageDtype
from .vector_quantizers import VectorQuantizer, GumbelVectorQuantizer
from .autoencoder import Encoder, Decoder
from .loss import VQLPIPS
from .datamodules import ImageDataModule
from .base_vae import BaseVQVAE


class VQVAE(BaseVQVAE, LightningModule):
    """
    A Vector Quantized Variational Autoencoder.
    """

    @dataclass
    class Config:
        """
        Configuration for the VQVAE.
        """

        learning_rate: float
        betas: List[float]
        eps: float
        weight_decay: float
        warmup_epochs: int
        decay_epochs: int
        reinit_every_n_epochs: int

        loss_type: str  # one of "mse", "lpips", "adversarial"
        init_cb: bool
        cb_size: int
        latent_dim: int
        commitment_cost: float

        # Autoencoder parameters
        channels: int
        num_res_blocks: int
        channel_multipliers: List[int]
        final_conv_channels: int

        image_size: int = 256

        # LPIPS parameters
        l1_weight: float = None
        l2_weight: float = None
        perc_weight: float = None

        # Experiemnt tracking
        track: bool = False

    def __init__(self, config: Config):
        super().__init__(image_size=config.image_size)
        self.config = config
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.quantizer = self._create_quantizer()
        self.criterion = self._create_loss()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, C, H, W) input tensor
        :return reconstructions: (B, C, H, W), quantizer_loss: float, used_indices: (B, S)
        """
        z = self.encoder(x)
        quantized, used_indices, e_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)

        return x_recon, e_loss, used_indices

    def training_step(self, batch, batch_index: int):
        """
        :param batch: images B C H W, or tuple if ffcv loader
        :param batch_index: used for logging reconstructions only once per epoch
        """
        images = self.preprocess_batch(batch[0] if isinstance(batch, tuple) else batch)
        x_recon, q_loss, used_indices = self.forward(images)

        # log reconstructions (every 5 epochs, for one batch)
        if batch_index == 2 and self.current_epoch % 5 == 0:
            if self.config.track:
                self.log_reconstructions(images, x_recon, t_or_v="t")

        if isinstance(self.criterion, VQLPIPS):
            loss, l1_loss, l2_loss, p_loss = self.criterion(q_loss, images, x_recon)
            g_loss, d_loss = torch.zeros(1), torch.zeros(1)
            g_weight, r1_penalty = 0.0, 0.0

        else:
            l2_loss = self.criterion(x_recon, images)
            l1_loss, g_loss, p_loss, d_loss = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            g_weight, r1_penalty = 0.0, 0.0
            loss = q_loss + l2_loss

        if True:
            self.log("g_weight", g_weight, sync_dist=True, on_step=False, on_epoch=True)
            self.log(
                "r1_penalty", r1_penalty, sync_dist=True, on_step=False, on_epoch=True
            )

            self.log(
                "train/loss",
                loss.detach().cpu().item(),
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/l1_loss",
                l1_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/l2_loss",
                l2_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/quant_loss",
                q_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/perc_loss",
                p_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/gen_loss",
                g_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/disc_loss",
                d_loss.detach().cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

        # batch index count (use non-deterministic for this operation)
        torch.use_deterministic_algorithms(False)
        used_indices = torch.bincount(
            used_indices.view(-1), minlength=self.config.cb_size
        )
        torch.use_deterministic_algorithms(True)

        self.train_epoch_usage_count = (
            used_indices if self.train_epoch_usage_count is None else +used_indices
        )

        return loss

    def on_train_start(self):
        """
        Initialize warmup or decay of the learning rate (if specified).
        Initialize const warmup and decay if using Gumbel Softmax.
        """
        # init warmup/decay lr
        lr = float(self.config.learning_rate)
        if (
            self.config.warmup_epochs is not None
            and self.config.decay_epochs is not None
        ):

            warmup_step_start = 0
            warmup_step_end = (
                self.config.warmup_epoch * self.trainer.num_training_batches
            )
            decay_step_end = self.config.decay_epoch * self.trainer.num_training_batches
            self.scheduler = LinearCosineScheduler(
                warmup_step_start, decay_step_end, lr, lr / 10, warmup_step_end
            )

        elif self.config.warmup_epochs is not None:

            warmup_step_start = 0
            warmup_step_end = (
                self.config.warmup_epochs * self.trainer.num_training_batches
            )
            self.scheduler = LinearScheduler(
                warmup_step_start, warmup_step_end, 1e-20, lr
            )

        elif self.config.decay_epochs is not None:

            decay_step_start = 0
            decay_step_end = (
                self.config.decay_epochs * self.trainer.num_training_batches
            )
            self.scheduler = CosineScheduler(
                decay_step_start, decay_step_end, lr, lr / 10
            )

        # if quantizer is gumbel
        if isinstance(self.quantizer, GumbelVectorQuantizer):
            temp, kl = self.quantizer.get_consts()
            if self.config.kl_warmup_epochs is not None:
                kl_start = 0
                kl_stop = int(self.kl_warmup_epochs * self.trainer.num_training_batches)
                self.quantizer.kl_warmup = CosineScheduler(kl_start, kl_stop, 0.0, kl)

            if (
                self.config.temp_decay_epochs is not None
                and self.temp_final is not None
            ):
                temp_start = 0
                temp_stop = int(
                    self.config.temp_decay_epochs * self.trainer.num_training_batches
                )
                self.quantizer.temp_decay = CosineScheduler(
                    temp_start, temp_stop, temp, self.temp_final
                )

    def on_train_batch_start(self, _, batch_index: int):
        """
        Update lr and gumbel quant values according to current epoch/batch index
        """
        current_step = (
            self.current_epoch * self.trainer.num_training_batches
        ) + batch_index

        # lr update
        if self.scheduler is not None:
            step_lr = self.scheduler.step(current_step)
        else:
            step_lr = self.config.learning_rate

        for optimizer in self.trainer.optimizers:
            for g in optimizer.param_groups:
                g["lr"] = step_lr

        # gumbel update and logging
        if isinstance(self.quantizer, GumbelVectorQuantizer):
            this_temp, this_kl = self.quantizer.get_consts()
            if self.quantizer.kl_warmup is not None:
                this_kl = self.quantizer.kl_warmup.step(current_step)
            if self.quantizer.temp_decay is not None:
                this_temp = self.quantizer.temp_decay.step(current_step)
            self.quantizer.set_consts(this_temp, this_kl)
        else:
            this_temp, this_kl = 0.0, 0.0

        if self.config.track:
            self.log("gumbel_quantizer/temperature", this_temp, sync_dist=True)
            self.log("gumbel_quantizer/kl_constant", this_kl, sync_dist=True)

    def on_train_epoch_end(self):

        if (
            self.config.reinit_every_n_epochs is not None
            and self.current_epoch % self.config.reinit_every_n_epochs == 0
            and self.current_epoch > 0
        ):
            self.quantizer.reinit_unused_codes(
                self.quantizer.get_codebook_usage(self.train_epoch_usage_count)[0]
            )

        self.train_epoch_usage_count = None

    def validation_step(self, batch: Any, batch_index: int):
        """
        :param batch: images B C H W, or tuple if ffcv loader
        :param batch_index: used for logging reconstructions only once per epoch
        """

        images = self.preprocess_batch(batch[0] if isinstance(batch, tuple) else batch)
        x_recon, q_loss, used_indices = self.forward(images)

        # log reconstructions (validation is done every 5 epochs by default)
        if batch_index == 2:
            if self.config.track:
                self.log_reconstructions(images, x_recon, t_or_v="v")

        if isinstance(self.criterion, VQLPIPS):
            loss, l1_loss, l2_loss, p_loss = self.criterion(q_loss, images, x_recon)
            g_loss, d_loss = torch.zeros(1), torch.zeros(1)

        else:
            l2_loss = self.criterion(x_recon, images)
            l1_loss, g_loss, p_loss, d_loss = (
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
            loss = q_loss + l2_loss

        if True or self.config.track:
            self.log(
                "validation/loss",
                loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/l1_loss",
                l1_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/l2_loss",
                l2_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/quant_loss",
                q_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/perc_loss",
                p_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/gen_loss",
                g_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "validation/disc_loss",
                d_loss.cpu().item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

        # batch index count (use non-deterministic for this operation)
        torch.use_deterministic_algorithms(False)
        used_indices = torch.bincount(
            used_indices.view(-1), minlength=self.config.cb_size
        )
        torch.use_deterministic_algorithms(True)

        self.val_epoch_usage_count = (
            used_indices if self.val_epoch_usage_count is None else +used_indices
        )
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log metrics on codebook usage
        """
        _, perplexity, cb_usage = self.quantizer.get_codebook_usage(
            self.val_epoch_usage_count
        )

        # log results
        if self.config.track:
            self.log(f"val_metrics/used_codebook", cb_usage, sync_dist=True)
            self.log(f"val_metrics/perplexity", perplexity, sync_dist=True)

        self.val_epoch_usage_count = None

        return

    def on_train_end(self):
        # ensure to destroy c++ scheduler object
        self.scheduler.destroy()

    def configure_optimizers(self):
        def split_decay_groups(
            named_modules: List,
            named_parameters: List,
            whitelist_weight_modules: Tuple[torch.nn.Module, ...],
            blacklist_weight_modules: Tuple[torch.nn.Module, ...],
            wd: float,
        ):
            """
            reference https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
            separate out all parameters to those that will and won't experience regularizing weight decay
            """

            decay = set()
            no_decay = set()
            for mn, m in named_modules:
                for pn, p in m.named_parameters():
                    fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(
                        m, whitelist_weight_modules
                    ):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(
                        m, blacklist_weight_modules
                    ):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in named_parameters}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert (
                len(inter_params) == 0
            ), "parameters %s made it into both decay/no_decay sets!" % (
                str(inter_params),
            )
            assert (
                len(param_dict.keys() - union_params) == 0
            ), "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),
            )

            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": wd,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0,
                },
            ]
            return optim_groups

        # parameters
        lr = float(self.config.learning_rate)
        betas = [float(b) for b in self.config.betas]
        eps = float(self.config.eps)
        weight_decay = float(self.config.weight_decay)

        # autoencoder optimizer
        ae_params = split_decay_groups(
            named_modules=list(self.encoder.named_modules())
            + list(self.decoder.named_modules())
            + list(self.quantizer.named_modules()),
            named_parameters=list(self.encoder.named_parameters())
            + list(self.decoder.named_parameters())
            + list(self.quantizer.named_parameters()),
            whitelist_weight_modules=(torch.nn.Conv2d,),
            blacklist_weight_modules=(torch.nn.GroupNorm, torch.nn.Embedding),
            wd=weight_decay,
        )
        ae_optimizer = torch.optim.AdamW(
            ae_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        return ae_optimizer

    @torch.no_grad()
    def log_reconstructions(self, ground_truths, reconstructions, t_or_v="t"):
        """
        log reconstructions
        """
        import wandb

        b = min(ground_truths.shape[0], 8)
        panel_name = "train" if t_or_v == "t" else "validation"

        display, _ = pack(
            [
                self.preprocess_visualization(ground_truths[:b]),
                self.preprocess_visualization(reconstructions[:b]),
            ],
            "* c h w",
        )

        display = make_grid(display, nrow=b)
        display = wandb.Image(display)
        self.logger.experiment.log({f"{panel_name}/reconstructions": display})

    def get_tokens(self, images: torch.Tensor) -> torch.IntTensor:
        """
        :param images: B, 3, H, W in range 0__1
        :return B, S batch of codebook indices
        """

        images = self.preprocess_batch(images)
        return self.quantizer.vec_to_codes(self.encoder(images))

    def quantize(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W in range 0__1
        :return B, S, D batch of quantized
        """
        images = self.preprocess_batch(images)
        return rearrange(
            self.quantizer(self.encoder(images))[0], "b d h w -> b (h w) d"
        )

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W in range 0__1
        :return reconstructions (B, 3, H, W)  in range 0__1
        """

        images = self.preprocess_batch(images)
        return self.preprocess_visualization(self(images)[0])

    def reconstruct_from_tokens(self, tokens: torch.IntTensor) -> torch.Tensor:
        """
        :param tokens: B, S where S is the sequence len
        :return (B, 3, H, W) reconstructed images in range 0__1
        """
        return self.preprocess_visualization(
            self.decoder(self.quantizer.codes_to_vec(tokens))
        )

    def on_test_epoch_start(self):

        # metrics for testing
        self.test_mse = MeanSquaredError().to("cuda")
        self.test_ssim = StructuralSimilarityIndexMeasure().to("cuda")
        self.test_psnr = PeakSignalNoiseRatio().to("cuda")
        self.test_rfid = FrechetInceptionDistance().to("cuda")

        # test used codebook, perplexity
        self.test_usage_count = None

    def test_step(self, images, _):

        # get reconstructions, used_indices
        images = images[0] if isinstance(images, tuple) else images
        reconstructions, _, used_indices = self.forward(self.preprocess_batch(images))
        reconstructions = self.preprocess_visualization(reconstructions)

        # batch index count (use non-deterministic for this operation)
        torch.use_deterministic_algorithms(False)
        used_indices = torch.bincount(
            used_indices.view(-1), minlength=self.config.cb_size
        )
        torch.use_deterministic_algorithms(True)

        self.test_usage_count = (
            used_indices if self.test_usage_count is None else +used_indices
        )

        # plot reconstruction (just for sanity check)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.float32(images[0].permute(1, 2, 0).cpu().numpy()))
        # ax[1].imshow(np.float32(reconstructions[0].permute(1, 2, 0).cpu().numpy()))
        # plt.show()

        # computed Metrics:

        # MSE
        self.test_mse.update(reconstructions, images)

        # SSIM
        self.test_ssim.update(reconstructions, images)

        # PSNR
        self.test_psnr.update(reconstructions, images)

        # rFID take uint 8
        conv = ConvertImageDtype(torch.uint8)
        reconstructions = conv(reconstructions)
        images = conv(images)

        # rFID
        self.test_rfid.update(reconstructions, real=False)
        self.test_rfid.update(images, real=True)

    def on_test_epoch_end(self):

        total_mse = self.test_mse.compute()
        self.log(f"mse", total_mse)

        total_ssim = self.test_ssim.compute()
        self.log(f"ssim", total_ssim)

        total_psnr = self.test_psnr.compute()
        self.log(f"psnr", total_psnr)

        total_fid = self.test_rfid.compute()
        self.log(f"rfid", total_fid)

        _, perplexity, cb_usage = self.quantizer.get_codebook_usage(
            self.test_usage_count
        )

        # log results
        self.log(f"used_codebook", cb_usage)
        self.log(f"perplexity", perplexity)

    def _create_quantizer(self):
        quantizer = VectorQuantizer(
            num_embeddings=self.config.cb_size,
            embedding_dim=self.config.latent_dim,
            commitment_cost=self.config.commitment_cost,  # float(q_conf["params"]["commitment_cost"])
        )
        if self.config.init_cb:
            quantizer.init_codebook()
        return quantizer

    def _create_encoder(self):
        channels = self.config.channels
        num_res_blocks = self.config.num_res_blocks
        channel_multipliers = self.config.channel_multipliers
        # final_conv_channels = (
        #     self.cb_size if q_conf["type"] == "gumbel" else self.latent_dim
        # )
        final_conv_channels = self.config.latent_dim
        return Encoder(
            channels, num_res_blocks, channel_multipliers, final_conv_channels
        )

    def _create_decoder(self):
        return Decoder(
            self.config.channels,
            self.config.num_res_blocks,
            self.config.channel_multipliers,
            self.config.latent_dim,
        )

    def _create_loss(self):
        if self.config.loss_type == "mse":
            return torch.nn.MSELoss()
        elif self.config.loss_type == "lpips":
            return VQLPIPS(
                self.config.l1_weight, self.config.l2_weight, self.config.perc_weight
            )
        else:
            raise ValueError(f"Invalid loss type: {self.config.loss_type}")


class VQVAETrainingPipeline:
    """
    A training pipeline for Vector Quantized Variational Autoencoders.
    """

    @dataclass
    class Config:
        """
        Configuration for the VQVAE training pipeline.
        """

        dataset_path: str
        image_size: int = 256
        # The number of nodes in the distributed training setup.
        num_nodes: int = 1
        workers: int = 1
        # The number of training epochs.
        num_epochs: int = 10
        # The batch size for training.
        batch_size: int = 16
        cumulative_bs: int = 16
        # The learning rate for the optimizer.
        learning_rate: float = 0.0001
        # The weight decay for the optimizer.
        # The path to the model checkpoint.
        model_checkpoint_path: str = None

        track: bool = False
        wandb_project: str = "vqvae"
        run_name: str = "vqvae_run"
        save_path: str = "logs/"
        save_every_n_epochs: int = 10

        seed: int = 0
        resume_from: str = None
        set_matmul_precision_for_a100: bool = False

    def fit(self, dataset_builder=None):
        import wandb

        # only for A100
        if self.config.set_matmul_precision_for_a100:
            self._set_matmul_precision()

        # configuration params (assumes some env variables in case of multi-node setup)
        gpus = torch.cuda.device_count()
        num_nodes = self.config.num_nodes
        rank = int(os.getenv("NODE_RANK")) if os.getenv("NODE_RANK") is not None else 0
        is_dist = gpus > 1 or num_nodes > 1

        workers = int(self.config.workers)
        seed = int(self.config.seed)

        cumulative_batch_size = int(self.config.cumulative_bs)
        batch_size_per_device = cumulative_batch_size // (num_nodes * gpus)

        base_learning_rate = float(self.config.learning_rate)
        learning_rate = base_learning_rate * math.sqrt(cumulative_batch_size / 256)

        max_epochs = int(self.config.num_epochs)

        pl.seed_everything(seed, workers=True)

        # logging stuff, checkpointing and resume
        log_to_wandb = self.config.track
        project_name = self.config.wandb_project
        # wandb_id = args.wandb_id

        run_name = str(self.config.run_name)
        save_checkpoint_dir = f"{self.config.save_path}{run_name}/"
        save_every_n_epochs = int(self.config.save_every_n_epochs)

        load_checkpoint_path = self.config.resume_from
        resume = load_checkpoint_path is not None

        logger = None
        if rank == 0:  # prevents from logging multiple times
            if self.config.track:
                logger = WandbLogger(
                    project=project_name,
                    name=run_name,
                    offline=not log_to_wandb,
                    # id=wandb_id,
                    resume="must" if resume else None,
                )
        else:
            # logger = WandbLogger(project=project_name, name=run_name, offline=True)
            pass

        # model params
        image_size = self.config.image_size
        ae_conf = {
            "channels": 64,
            "num_res_blocks": 2,
            "channel_multipliers": [1, 2, 4, 8],
        }
        q_conf = {
            "num_embeddings": 1024,
            "embedding_dim": 256,
            "type": "standard",
            "params": {"commitment_cost": 0.25},
            "reinit_every_n_epochs": None,
        }
        l_conf = {
            "l1_weight": 0.1,
            "l2_weight": 1.0,
            "perc_weight": 0.1,
            "adversarial_params": {
                "start_epoch": 1,
                "loss_type": "non-saturating",
                "g_weight": 0.8,
                "use_adaptive": True,
                "r1_reg_weight": 10.0,
                "r1_reg_every": 16,
            },
        }
        t_conf = {
            "base_lr": base_learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "decay_epochs": 150,
            "max_epochs": max_epochs,
        }
        self.t_conf = t_conf
        self.l_conf = l_conf
        self.ae_conf = ae_conf
        self.q_conf = q_conf

        # check if using adversarial loss
        use_adversarial = (
            l_conf is not None
            and "adversarial_params" in l_conf.keys()
            and l_conf["adversarial_params"] is not None
        )

        # get model
        if resume:
            # TODO: this may not work..
            model = VQVAE.load_from_checkpoint(
                load_checkpoint_path,
                strict=False,
                image_size=image_size,
                ae_conf=ae_conf,
                q_conf=q_conf,
                l_conf=l_conf,
                t_conf=t_conf,
                init_cb=False,
                load_loss=True,
            )
        else:
            model = VQVAE(
                config=VQVAE.Config(
                    learning_rate=learning_rate,
                    betas=t_conf["betas"],
                    eps=t_conf["eps"],
                    weight_decay=t_conf["weight_decay"],
                    warmup_epochs=None,
                    decay_epochs=t_conf["decay_epochs"],
                    reinit_every_n_epochs=q_conf["reinit_every_n_epochs"],
                    image_size=image_size,
                    loss_type="mse",
                    init_cb=True,
                    cb_size=q_conf["num_embeddings"],
                    latent_dim=q_conf["embedding_dim"],
                    commitment_cost=q_conf["params"]["commitment_cost"],
                    channels=ae_conf["channels"],
                    num_res_blocks=ae_conf["num_res_blocks"],
                    channel_multipliers=ae_conf["channel_multipliers"],
                    final_conv_channels=3,
                    l1_weight=l_conf["l1_weight"],
                    l2_weight=l_conf["l2_weight"],
                    perc_weight=l_conf["perc_weight"],
                    track=log_to_wandb,
                )
            )

        # data loading (standard pytorch lightning or ffcv)
        datamodule = self._create_datamodule(
            "standard",
            self.config.dataset_path,
            image_size,
            batch_size_per_device,
            workers,
            seed,
            is_dist,
            mode="train",
        )

        # callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_checkpoint_dir,
            filename="{epoch:02d}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=save_every_n_epochs,
        )

        callbacks = [LearningRateMonitor(), checkpoint_callback]

        # trainer
        # set find unused parameters if using vqgan (adversarial training)
        trainer = pl.Trainer(
            strategy=DDPStrategy(
                find_unused_parameters=use_adversarial, static_graph=not use_adversarial
            ),
            accelerator="gpu",
            num_nodes=num_nodes,
            devices=gpus,
            precision="16-mixed",
            callbacks=callbacks,
            deterministic=True,
            logger=logger,
            max_epochs=max_epochs,
            check_val_every_n_epoch=5,
        )

        print(f"[INFO] workers: {workers}")
        print(f"[INFO] batch size per device: {batch_size_per_device}")
        print(f"[INFO] cumulative batch size (all devices): {cumulative_batch_size}")
        print(f"[INFO] final learning rate: {learning_rate}")

        # check to prevent later error
        if use_adversarial and batch_size_per_device % 4 != 0:
            raise RuntimeError(
                "batch size per device must be divisible by 4! (due to stylegan discriminator forward pass)"
            )

        print(
            f"Model trainable parameters: {sum(p.numel() for p in model.parameters())}"
        )
        print(
            f"Fitting on data module: {datamodule}. Train size: {datamodule.train_dataloader().dataset.__len__()}"
        )
        print(f"Val size: {datamodule.val_dataloader().dataset.__len__()}")
        # trainer.fit(model, datamodule, ckpt_path=load_checkpoint_path)
        trainer.fit(
            model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=load_checkpoint_path,
        )

        # ensure wandb has stopped logging
        if log_to_wandb:
            wandb.finish()

    def __init__(self, config: Config):
        self.config = config

    def _create_model(self):
        pass

    def _create_optimizer(self):
        pass

    def _create_scheduler(self):
        pass

    def _create_datamodule(
        self,
        loader_type: str,
        dirpath: str,
        image_size: int,
        batch_size: int,
        workers: int,
        seed: int,
        is_dist: bool,
        mode: str = "train",
    ):
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"dataset path not found: {dirpath}")

        else:

            if loader_type == "standard":

                if mode == "train":
                    train_folder = f"{dirpath}train/"
                    val_folder = f"{dirpath}validation/"
                    print("Creating image datamodule for training")
                    datamodule = ImageDataModule(
                        image_size, batch_size, workers, train_folder, val_folder
                    )
                    datamodule.setup(stage="fit")
                    return datamodule
                else:
                    test_folder = f"{dirpath}test/"
                    datamodule = ImageDataModule(
                        image_size, batch_size, workers, test_folder=test_folder
                    )
                    datamodule.setup(stage="test")
                    return datamodule

            else:
                raise ValueError(f"loader type not recognized: {loader_type}")

    def _create_trainer(self):
        pass

    def _set_matmul_precision(self):
        import os

        """
        If using Ampere Gpus enable using tensor cores.
        Don't know exactly which other devices can benefit from this, but torch should throw a warning in case.
        Docs: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        """

        gpu_cores = os.popen("nvidia-smi -L").readlines()[0]

        if "A100" in gpu_cores:
            torch.set_float32_matmul_precision("high")
            print('[INFO] set matmul precision "high"')


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES='0' python -m cvlization.torch.net.vae_resnet.vqvae | tee tmp.log

    Did not check the quality of the model yet.
    """
    config = VQVAETrainingPipeline.Config(
        dataset_path="data/tmp/",
        num_nodes=1,
        workers=1,
        num_epochs=10,
        batch_size=16,
        cumulative_bs=16,
        learning_rate=0.0001,
        model_checkpoint_path=None,
        track=False,
        wandb_project="vqvae",
        run_name="vqvae_run",
        save_path="logs/",
        save_every_n_epochs=10,
        seed=0,
        resume_from=None,
        set_matmul_precision_for_a100=False,
    )

    pipeline = VQVAETrainingPipeline(config)
    pipeline.fit()
