import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from masked_diffusion import (
    create_diffusion,
)
from diffusers.models import AutoencoderKL
# from adan import Adan
from torch.distributed.optim import ZeroRedundancyOptimizer
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scale_factor=0.18215, # scale_factor follows DiT and stable diffusion.
        opt_type='adamw',
        use_zero=False,
        data_is_latent=False,  # If True, data is already latent. No need to run encoder.
        track=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.scale_factor = scale_factor
        self.data_is_latent = data_is_latent
        self.track = track

        self.step = 0
        self.resume_step = 0
        # TODO: support multi-GPU training.
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        if opt_type=='adamw':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=AdamW,
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )
            else:
                self.opt = AdamW(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
                )
        elif opt_type=='adan':
            if use_zero:
                self.opt = ZeroRedundancyOptimizer(
                    self.mp_trainer.master_params,
                    optimizer_class=Adan,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    max_grad_norm=1, fused=True
                )
            else:
                self.opt = Adan(
                    self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1, fused=True)

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = False
            self.ddp_model = model
            # self.ddp_model = DDP(
            #     self.model,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            #     find_unused_parameters=False,
            # )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.instantiate_first_stage()


    def instantiate_first_stage(self):
        model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dist_util.dev())
        model = th.compile(model)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # https://github.com/huggingface/diffusers/blob/29b2c93c9005c87f8f04b1f0835babbcea736204/src/diffusers/models/autoencoder_kl.py
    @th.no_grad()
    def get_first_stage_encoding(self, x):
            encoder_posterior = self.first_stage_model.encode(x, return_dict=True)[0]

            z = encoder_posterior.sample()
            return z.to(dist_util.dev()) * self.scale_factor

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                import torch

                self.model.load_state_dict(
                    torch.load(resume_checkpoint)
                    # dist_util.load_state_dict(
                    #     resume_checkpoint, map_location=dist_util.dev()
                    # )
                )
        # TODO: support multi-GPU training.
        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch = next(self.data)
            if isinstance(batch, list):
                # print(batch[0]); import sys; sys.exit()
                batch = batch[0]
                cond = {}
            elif isinstance(batch, tuple) and len(batch) == 2:
                batch, cond = next(self.data)
            else:
                raise ValueError(f"Data loader must return either 1 or 2 tensors. Got {{batch}}, of type {type(batch)}.")
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                if self.track:
                    import wandb
                    CURRENT = logger.get_current()
                    to_log = {}
                    for k, v in CURRENT.name2val.items():
                        to_log[k] = v
                    # for k, v in CURRENT.name2cnt.items():
                    #     to_log[k] = v
                    wandb.log(to_log, step=self.step)

                logger.dumpkvs()

            sample_interval = self.log_interval * 10
            if self.step % sample_interval == 0:
                if not hasattr(self, "diffusion"):
                    self.diffusion = create_diffusion(
                        diffusion_steps=1000,
                        noise_schedule="linear",
                        timestep_respacing=""
                    )
                diffusion = self.diffusion
                model = self.model

                # sample, decode and log
                use_ddim = True
                sample_fn = (
                    diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
                )
                batch_size = 1
                latent_dim = 4
                latent_size = 32  # TODO: this is hardcoded for now.
                clip_denoised = False
                z = th.randn(batch_size, latent_dim, latent_size, latent_size, device=dist_util.dev())
                model_kwargs = {"y": None}
                sample = sample_fn(
                    model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=clip_denoised,
                    progress=True, 
                    model_kwargs=model_kwargs,
                    device=dist_util.dev()
                )
                vae = self.first_stage_model
                sample = vae.decode(sample / 0.18215).sample
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # clip in range -1,1
                sample = sample.permute(0, 2, 3, 1)  # NHWC
                sample = sample.contiguous()
                print("sampled image:", sample.shape)
                if self.track:
                    import wandb
                    wandb.log({"sample/generated_decoded": wandb.Image(sample[0].cpu().numpy())}, step=self.step)

            if (self.step + 1) % self.save_interval == 0:
                if hasattr(self.opt, "consolidate_state_dict"):
                    self.opt.consolidate_state_dict()
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):

            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            if self.data_is_latent:
                micro = micro * self.scale_factor
            else:
                micro = self.get_first_stage_encoding(micro).detach()
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            micro_cond_mask = micro_cond.copy()
            micro_cond_mask['enable_mask']=True
            compute_losses_mask = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond_mask,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
                losses_mask = compute_losses_mask()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
                    losses_mask = compute_losses_mask()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach() + losses_mask["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + (losses_mask["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            log_loss_dict(
                self.diffusion, t, {'m_'+k: v * weights for k, v in losses_mask.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # TODO: support multi-GPU training.
            if True or dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if True or dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)