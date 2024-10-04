# pip install diffusers[training] accelerate datasets
# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
# TODO: the name `unet` is no longer correct, other architectures can be used.

import argparse
from types import SimpleNamespace
import inspect
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
import wandb
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from cvlization.torch.training_pipeline.dit.dit import DiT


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@dataclass
class Trainer:
    model: torch.nn.Module
    output_dir: str
    noise_scheduler: DDPMScheduler
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    accelerator: Optional[Accelerator] = None
    logger: str = "tensorboard"  # tensorboard or wandb
    ema_model: Optional[EMAModel] = None
    use_ema: bool = True
    num_epochs: int = 10
    train_batch_size: int = 32
    total_batch_size: int = 32
    ddpm_num_inference_steps: int = 10
    gradient_accumulation_steps: int = 1
    num_update_steps_per_epoch: int = 1000
    prediction_type: str = "epsilon" # epsilon or sample
    eval_batch_size: int = 32
    checkpointing_steps: int = 5000
    save_images_epochs: int = 1
    save_model_epochs: int = 5
    max_train_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    def train(self, train_dataloader):
        logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {self.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint != "latest":
                path = os.path.basename(self.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.gradient_accumulation_steps
                first_epoch = global_step // self.num_update_steps_per_epoch
                resume_step = resume_global_step % (self.num_update_steps_per_epoch * self.gradient_accumulation_steps)

        # Train!
        model = self.model
        ema_model = self.ema_model
        noise_scheduler = self.noise_scheduler
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        accelerator = self.accelerator
        for epoch in range(first_epoch, self.num_epochs):
            model.train()
            progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(train_dataloader):
                # Skip steps after 10, just for testing
                # if step > 10:
                #     break

                # Skip steps until we reach the resumed step
                if self.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % self.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                clean_images = batch["input"]
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    model_output = model(noisy_images, timesteps)
                    if hasattr(model_output, "sample"):
                        model_output = model_output.sample

                    if self.prediction_type == "epsilon":
                        assert model_output.shape == noise.shape, f"Model output shape {model_output.shape} != noise shape {noise.shape}"
                        loss = F.mse_loss(model_output, noise)  # this could have different weights!
                    elif self.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        loss = snr_weights * F.mse_loss(
                            model_output, clean_images, reduction="none"
                        )  # use SNR weighting from distillation paper
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.use_ema:
                        ema_model.step(model.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                if self.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            progress_bar.close()

            accelerator.wait_for_everyone()

            # Generate sample images for visual inspection
            if accelerator.is_main_process:
                if epoch % self.save_images_epochs == 0 or epoch == self.num_epochs - 1:
                    unet = accelerator.unwrap_model(model)

                    if self.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    # Prevent an error that complained about the DDPMPipeline does not have .device.
                    if not hasattr(unet, "device"):
                        unet.device = accelerator.device

                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )

                    generator = torch.Generator(device=pipeline.device).manual_seed(0)
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        batch_size=self.eval_batch_size,
                        num_inference_steps=self.ddpm_num_inference_steps,
                        output_type="numpy",
                    ).images

                    if self.use_ema:
                        ema_model.restore(unet.parameters())

                    # denormalize the images and save to tensorboard
                    images_processed = (images * 255).round().astype("uint8")

                    if self.logger == "tensorboard":
                        if is_accelerate_version(">=", "0.17.0.dev0"):
                            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                        else:
                            tracker = accelerator.get_tracker("tensorboard")
                        tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                    elif self.logger == "wandb":
                        # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                        accelerator.get_tracker("wandb").log(
                            {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                            step=global_step,
                        )

                if epoch % self.save_model_epochs == 0 or epoch == self.num_epochs - 1:
                    # save the model
                    unet = accelerator.unwrap_model(model)

                    if self.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                    # Save the pretrained huggingface pipeline
                    pipeline.save_pretrained(self.output_dir)

                    if self.use_ema:
                        ema_model.restore(unet.parameters())


        accelerator.end_training()


@dataclass
class TrainingPipeline:
    args: argparse.Namespace

    def fit(self, dataset_builder):
        args = self.args
        self._configure_logging()
        model = self._create_model()
        ema_model = self._create_ema_model(model)
        self._create_scheduler()
        accelerator = self._prepare_accelerator(ema_model)
        train_dataloader = self._create_train_dataloader(dataset_builder.training_dataset())
        optimizer, lr_scheduler = self._create_optimizer(model=model, train_dataloader=train_dataloader)
        
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        ema_model.to(accelerator.device)

        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model

        trainer = self._create_trainer()
        trainer.train(train_dataloader=train_dataloader)


    def _configure_logging(self):
        # Make one log on every process with the configuration for debugging.
        args = self.args
        self._logging_dir = os.path.join(args.output_dir, args.logging_dir)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        if args.logger == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

        elif args.logger == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb
    
    def _create_trainer(self):
        args = self.args
        accelerator = self._accelerator
        train_dataloader = self.train_dataloader
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.num_epochs * num_update_steps_per_epoch
        trainer = Trainer(
            model=self.model,
            noise_scheduler=self.noise_scheduler,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            accelerator=self._accelerator,
            ema_model=self.ema_model,
            logger=args.logger,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            save_model_epochs=args.save_model_epochs,
            resume_from_checkpoint=args.resume_from_checkpoint,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_batch_size=total_batch_size,
            ddpm_num_inference_steps=args.ddpm_num_inference_steps,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            checkpointing_steps=args.checkpointing_steps,
            max_train_steps=max_train_steps,
            eval_batch_size=args.eval_batch_size,
            use_ema=True,
            prediction_type=args.prediction_type,
        )
        return trainer

    def _prepare_accelerator(self, ema_model):
        args = self.args
        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.logger,
            # logging_dir=self._logging_dir,
            project_dir=self._logging_dir,
            project_config=accelerator_project_config,
        )
        self._accelerator = accelerator
        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

            def load_model_hook(models, input_dir):
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)
        
        accelerator = self._accelerator
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            run = "Diffuser Unconditional"
            accelerator.init_trackers(run)
        
        
        return accelerator

    def _create_ema_model(self, model):
        # Create EMA for the model.
        args = self.args
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config if hasattr(model, "config") else None,
        )
        self.ema_model = ema_model
        return ema_model
    
    def _create_scheduler(self):
        # Initialize the scheduler
        args = self.args
        accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
        if accepts_prediction_type:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                prediction_type=args.prediction_type,
            )
        else:
            noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
        self.noise_scheduler = noise_scheduler
        return noise_scheduler

    def _create_model(self):
        args = self.args
        if args.model_config_name_or_path is None:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        elif args.model_config_name_or_path == "dit_t0":
            model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=3,
                learn_sigma=False,
                hidden_size=32,
                mlp_ratio=2,
                depth=3,
                num_heads=1,
                class_dropout_prob=0.1,
            )
        elif args.model_config_name_or_path == "dit_t1":
            model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=3,
                learn_sigma=False,
                hidden_size=32 * 6,
                mlp_ratio=2,
                depth=3,
                num_heads=6,
                class_dropout_prob=0.1,
            )
        elif args.model_config_name_or_path == "dit_t2":
            model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=3,
                learn_sigma=False,
                hidden_size=32,
                mlp_ratio=2,
                depth=12,
                num_heads=1,
                class_dropout_prob=0.1,
            )
        elif args.model_config_name_or_path == "dit_t3":
            model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=3,
                learn_sigma=False,
                hidden_size=32 * 6,
                mlp_ratio=2,
                depth=12,
                num_heads=6,
                class_dropout_prob=0.1,
            )
        elif args.model_config_name_or_path == "dit_t4":
            model = DiT(
                input_size=32,
                patch_size=2,
                in_channels=3,
                learn_sigma=False,
                hidden_size=32 * 6,
                mlp_ratio=4,
                depth=12,
                num_heads=6,
                class_dropout_prob=0.1,
            )
        elif args.model_config_name_or_path == "dit_s_2":
            model = DiT(
                depth=12,
                in_channels=3,
                hidden_size=384,
                patch_size=2,
                num_heads=6,
                learn_sigma=False,
            )
        elif args.model_config_name_or_path == "dit_s_4":
            model = DiT(
                depth=12,
                in_channels=3,
                hidden_size=384,
                patch_size=4,
                num_heads=6,
                learn_sigma=False,
            )
        elif args.model_config_name_or_path == "dit_b_2":
            model = DiT(
                depth=12,
                in_channels=3,
                hidden_size=768,
                patch_size=2,
                num_heads=12,
                learn_sigma=False,
            )
        elif args.model_config_name_or_path == "dit_b_4":
            model = DiT(
                depth=12,
                in_channels=3,
                hidden_size=768,
                patch_size=4,
                num_heads=12,
                learn_sigma=False,
            )
        elif args.model_config_name_or_path == "dit_l_2":
            model = DiT(
                depth=24,
                in_channels=3,
                hidden_size=1024,
                patch_size=2,
                num_heads=16,
                learn_sigma=False,
            )
        elif args.model_config_name_or_path == "dit_l_4":
            model = DiT(
                depth=24,
                in_channels=3,
                hidden_size=1024,
                patch_size=4,
                num_heads=16,
                learn_sigma=False,
            )
            
        else:
            config = UNet2DModel.load_config(args.model_config_name_or_path)
            model = UNet2DModel.from_config(config)

        if not hasattr(model, "config"):
            model.config = SimpleNamespace(
                sample_size=args.resolution,
                in_channels=3,
            )
            
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            
        self.model = model
        return model
        
    def _create_optimizer(self, model, train_dataloader):
        args = self.args
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        self.optimizer = optimizer

        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )
        self.lr_scheduler = lr_scheduler

        return optimizer, lr_scheduler

    def _augument_dataset(self, dataset):
        args = self.args
        augmentations = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def transform_images(examples):
            if isinstance(examples, dict):
                if "image" in examples:
                    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
                elif "img" in examples:
                    images = [augmentations(image.convert("RGB")) for image in examples["img"]]
                else:
                    raise ValueError("No image key found in the dataset. Got keys: {}".format(examples.keys()))
            else:
                images = [augmentations(image.convert("RGB")) for image in examples]
            return {"input": images}
    
        dataset.set_transform(transform_images)
        return dataset
    
    def _create_train_dataloader(self, dataset):
        args = self.args
        dataset = self._augument_dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )
        return train_dataloader


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



def main(args):

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    
    training_pipeline = TrainingPipeline(args)
    class DatasetBuilder:
        def training_dataset(self):
            return dataset
        
    training_pipeline.fit(DatasetBuilder())



if __name__ == "__main__":
    from .argparser import parse_args
    args = parse_args()
    main(args)