"""
Adapted from https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/iddpm/__init__.py

TODO:
- estimate MFU
"""

from functools import partial
import torch
import wandb
from einops import rearrange
import iddpm_scheduler.gaussian_diffusion as gd
from iddpm_scheduler.respace import SpacedDiffusion, space_timesteps


class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON
                if not predict_xstart
                else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale

    def sample_unconditional(
        self,
        n_samples: int,
        model,
        z_size,
        device,
        additional_args=None,
    ):
        z = torch.randn(n_samples, *z_size, device=device)
        # forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.p_sample_loop(
            # forward,
            model,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=additional_args,
            progress=True,
            device=device,
        )
        return samples

    def sample(
        self,
        model,
        text_encoder,
        z_size,
        prompts,
        device,
        additional_args=None,
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        z = torch.cat([z, z], 0)
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples


# def forward(model, x, timestep, **kwargs):
#     return model(x, timestep, **kwargs)


def forward_with_cfg(model, x, timestep, y, cfg_scale, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--track", action="store_true", help="Track the experiment in W&B"
    )
    parser.add_argument(
        "--project", type=str, default="flying_mnist", help="W&B project name"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Max training steps"
    )
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--checkpoint_every", type=int, default=1000, help="Checkpoint every N steps"
    )
    parser.add_argument("--depth", type=int, default=6, help="Depth of the model")
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of the model"
    )
    parser.add_argument(
        "--patch_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default=(1, 2, 2),
        help="Patch size",
    )
    parser.add_argument(
        "--num_heads", type=int, default=3, help="Number of attention heads"
    )
    parser.add_argument(
        "--sample_every", type=int, default=100, help="Sample every N steps"
    )
    parser.add_argument(
        "--diffusion_steps", type=int, default=1000, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients every N steps",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate of the optimizer"
    )
    parser.add_argument(
        "--clip_grad", type=float, default=1.0, help="Clip gradients. Set to None to disable"
    )
    parser.add_argument(
        "--latent_frames_to_generate",
        type=int,
        default=8,
        help="Number of latent frames to generate",
    )
    parser.add_argument(
        "--tokens_input_file",
        type=str,
        default=None, # "flying_mnist_tokens_32frames_train.npy",
        help="Path to the tokenized input file",
    )
    parser.add_argument(
        "--latents_input_file",
        type=str,
        default=None, # "data/latents/flying_mnist__model-nilqq143_latents_32frames_train.npy",
        help="Path to the latents input file",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="zzsi_kungfu/videogpt/model-kbu39ped:v11",
        help="VAE model name",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a denoiser model checkpoint",
    )
    parser.add_argument(
        "--enable_flashattn",
        action="store_true",
        help="Enable FlashAttention in the denoiser",
    )
    return parser.parse_args()


def train_on_latents(
    device="cuda",
    batch_size=2,
    max_steps=1000,
    log_every=10,
    sample_every=100,
    checkpoint_every=1000,
    depth=6,
    hidden_size=768,
    patch_size=(1, 2, 2),
    num_heads=3,
    diffusion_steps=1000,
    accumulate_grad_batches=1,
    lr=1e-4,
    latent_frames_to_generate=8,
    clip_grad=None,
    resume_from:str=None,
    vae:str = "zzsi_kungfu/videogpt/model-kbu39ped:v11",
    tokens_input_file:str = None, # "flying_mnist_tokens_32frames_train.npy",
    latents_input_file:str = None, # "data/latents/flying_mnist__model-nilqq143_latents_32frames_train.npy",
    enable_flashattn=False,
    track=False,
    **kwargs,
):
    import numpy as np
    from cvlization.torch.net.vae.video_vqvae import VQVAE
    from stdit.model import STDiT
    from uuid import uuid4

    local_run_id = uuid4().hex[:6]
    model_id = vae.split("/")[-1].split(":")[0]
    print(f"model_id: {model_id}")
    # load from numpy
    if latents_input_file is not None:
        latents = np.load(latents_input_file, mmap_mode="r")
        assert model_id in latents_input_file, f"Expected model_id {model_id} in {latents_input_file}"
        token_ids = None
        assert len(latents.shape) == 5, f"Expected 5D tensor, got {latents.shape}"
        print(f"Loaded latents from {latents_input_file}. Shape: {latents.shape}")
    elif tokens_input_file is not None:
        token_ids = np.load(tokens_input_file)
        latents = None
        assert len(token_ids.shape) == 4, f"Expected 4D tensor, got {token_ids.shape}"
        assert model_id in tokens_input_file, f"Expected model_id {model_id} in {tokens_input_file}"
    # convert token_ids to embeddings
    if ":" in vae:
        # It is a wandb model. Load it using VQVAE.from_pretrained
        vae = VQVAE.from_pretrained(vae)
    else:
        # It is a huggingface model. diffusers.models.AutoencoderKL
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(vae)
    vae.eval()
    vae.to(device)

    if latents is None and token_ids is not None:
        with torch.no_grad():
            # TODO: This loads all data into GPU memory! Use a dataloader instead
            token_ids = torch.tensor(token_ids.astype(np.int32), dtype=torch.long)
            # TODO: estimate these values automatically from z
            # latent_multiplier = 9
            # latent_bias = -0.27
            z = vae.to("cpu").vq.codes_to_vec(token_ids)
            assert len(z.shape) == 5, f"Expected 5D tensor, got {z.shape}"
            assert (
                z.shape[2] == token_ids.shape[1]
            ), f"Expected the temporal dimension has size {token_ids.shape[1]}, got {z.shape[2]}"
            # print(z.shape)  # (1000, 4, 8, 64, 64)
            latent_multiplier = 1 / z.std()
            latent_bias = -z.mean() * latent_multiplier
            orig_z = z
            z = z * latent_multiplier + latent_bias
    else:
        z = torch.tensor(latents, dtype=torch.float32)
        assert len(z.shape) == 5, f"Expected 5D tensor, got {z.shape}"
        # assert z.shape[2] == latent_frames_to_generate, f"Expected temporal dimension has size {latent_frames_to_generate}, got {z.shape[2]}"
        # Compute latent multiplier and bias to make the mean 0 and std 1
        latent_multiplier = 1 / z.std()
        latent_bias = -z.mean() * latent_multiplier
        orig_z = z
        z = z * latent_multiplier + latent_bias
    
    assert z.shape[1] == 4, f"Expected latent dimension of 4, got {z.shape[1]}"

    def get_batch(latent_frames_to_generate: int):
        idx = np.random.choice(len(z), batch_size, replace=False)
        if latent_frames_to_generate < z.shape[2]:
            j = np.random.choice(z.shape[2] - latent_frames_to_generate, 1)[0]
        else:
            j = 0
            latent_frames_to_generate = z.shape[2]
        batch_z = z[idx, :, j:(j + latent_frames_to_generate), :, :]
        assert len(batch_z.shape) == 5, f"Expected 5D tensor, got {batch_z.shape}. Shape of z is {z.shape}"
        assert batch_z.shape[2] == latent_frames_to_generate, f"Expected temporal dimension has size {latent_frames_to_generate}, got {batch_z.shape[2]}"
        return torch.Tensor(batch_z).to(device)

    # TODO: with flash attn, got RuntimeError: Input type (c10::Half) and bias type (float) should be the same
    denoiser = STDiT(
        # depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs
        # input_size=z.shape[2:],
        input_size=(latent_frames_to_generate, z.shape[3], z.shape[4]),
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
        unconditional=True,
        enable_flashattn=enable_flashattn,
        dtype=torch.float16 if enable_flashattn else torch.float32,
    ).to(device)
    if resume_from is not None:
        # If the model name in resume_from is a wandb model,
        # then first download the model.
        if ":" in resume_from:
            import wandb
            import os

            api = wandb.Api()
            # skip if the file already exists
            artifact_dir = f"artifacts/{resume_from.split('/')[-1]}"
            if os.path.exists(artifact_dir):
                print(f"Model already exists at {artifact_dir}")
            else:
                artifact_dir = api.artifact(resume_from).download()
            # The file is model.ckpt.
            state_dict = torch.load(artifact_dir + "/model.ckpt")
            trimmed_state_dict = state_dict["model"]
            # print(trimmed_state_dict.keys())
            for k in ["pos_embed_temporal"]:
                trimmed_state_dict.pop(k, None)
            denoiser.load_state_dict(trimmed_state_dict, strict=False)
        else:
            checkpoint = torch.load(resume_from)
            trimmed_state_dict = checkpoint["model"]
            for k in ["pos_embed_temporal"]:
                trimmed_state_dict.pop(k, None)
            denoiser.load_state_dict(trimmed_state_dict, strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # start_step = checkpoint["step"]
        print(f"Resuming from {resume_from}")
    print("Denoiser:")
    print(denoiser)
    num_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Number of parameters: {num_params / 1_000_000:.2f} Million")
    # save to wandb
    if track:
        import wandb

        wandb.config.update(
            {
                "num_params": num_params,
            }
        )
    diffusion = IDDPM(
        num_sampling_steps=diffusion_steps,
    )

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)

    # training loop
    grad_norm = None
    for i in range(max_steps):
        x = get_batch(latent_frames_to_generate)
        assert x.shape[2] == latent_frames_to_generate, f"Expected temporal dimension has size {latent_frames_to_generate}, got {x.shape[2]}"
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        # print("x:", x.shape, "t:", t.shape, "latent_frames_to_generate:", latent_frames_to_generate)
        loss_dict = diffusion.training_losses(
            model=denoiser, x_start=x, t=t, model_kwargs=None
        )

        # Backward & update
        loss = loss_dict["loss"].mean()
        (loss / accumulate_grad_batches).backward()

        if i == 0:
            # Decode the ground truth latents
            with torch.no_grad():
                orig_z_first = orig_z[:1].to(device)
                vae = vae.to(device)
                if isinstance(vae, VQVAE):
                    video = vae.decode(orig_z_first)
                else:
                    # It is a huggingface AutoencoderKL model
                    t = z.shape[2]
                    video = vae.decode(rearrange(orig_z_first, "b c t h w -> (b t) c h w"))
                    video = video.sample
                    video = rearrange(video, "(b t) c h w -> b c t h w", t=t)
            video = (video - video.min()) / (
                video.max() - video.min() + 1e-6
            )
            video = (video * 255).to(torch.uint8)
            video = rearrange(video, "b c t h w -> t c h (b w)")
            assert video.shape[1] == 3, f"shape of video is {video.shape}"
            
            if track:
                import wandb
                display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
                wandb.log({"sampled/ground_truth_decoded": display})

        if (i + 1) % accumulate_grad_batches == 0:
		    # Update Optimizer
            if clip_grad is not None:
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), clip_grad)

            optimizer.step()
            optimizer.zero_grad()

        # Callbacks

        if i % log_every == 0:
            print(f"Step {i}: {loss.item()}")
            if track:
                import wandb

                x_mean = x.mean()
                x_std = x.std()
                lr = optimizer.param_groups[0]["lr"]
                to_log = {
                    "train/loss": loss.item(),
                    "train/x_mean": x_mean,
                    "train/x_std": x_std,
                    "train/lr": lr,
                }
                if grad_norm is not None:
                    to_log["train/grad_norm"] = grad_norm.mean().item()
                wandb.log(to_log)
        
        if (i + 1) % checkpoint_every == 0:
            from pathlib import Path

            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(
                {
                    "model": denoiser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": i,
                },
                f"checkpoints/iddpm_{i}.ckpt",
            )
            if track:
                metadata = {"loss": loss.item(), "step": i}
                artifact = wandb.Artifact(
                    name="denoiser_model_{local_run_id}", type="model",
                    metadata=metadata
                )
                artifact.add_file(f"checkpoints/iddpm_{i}.ckpt", name=f"model.ckpt")
                wandb.log_artifact(artifact)

        if i % sample_every == 0:
            with torch.no_grad():
                samples = diffusion.sample_unconditional(
                    model=denoiser,
                    # text_encoder=None,
                    n_samples=1,
                    z_size=(z.shape[1], latent_frames_to_generate, z.shape[3], z.shape[4]),
                    # prompts=[],  # ["a", "b"],
                    device=device,
                    additional_args=None,
                )
                # print(samples.shape)
                # TODO: multiply a scaler factor to latents to make mean = 0, std = 1
                # decode z into a video
                assert samples.shape[2] == latent_frames_to_generate, f"Expected temporal dimension has size {latent_frames_to_generate}, got {samples.shape[2]}"
                assert samples.shape[3:] == z.shape[3:], f"shape of samples is {samples.shape}, shape of z is {z.shape}"
                sampled_z = (samples - latent_bias) / latent_multiplier
                if isinstance(vae, VQVAE):
                    video = vae.decode(sampled_z)
                else:
                    # It is a huggingface AutoencoderKL model
                    t = sampled_z.shape[2]
                    # sampled_z = sampled_z / 0.18
                    video = vae.decode(rearrange(sampled_z, "b c t h w -> (b t) c h w"))
                    video = video.sample
                    video = rearrange(video, "(b t) c h w -> b c t h w", t=t)
                video = (video - video.min()) / (video.max() - video.min() + 1e-6)
                video = (video * 255).to(torch.uint8)
                video = rearrange(video, "b c t h w -> t c h (b w)")
                assert video.shape[1] == 3, f"shape of video is {video.shape}"
                if track:
                    import wandb

                    display = wandb.Video(video.detach().cpu(), fps=5, format="mp4")
                    wandb.log(
                        {
                            "sampled/generated_video": display,
                        }
                    )


if __name__ == "__main__":
    args = get_args()
    if args.track:
        import wandb

        wandb.init(project=args.project)
        wandb.config.update(args)
    train_on_latents(**vars(args))
