"""
Adapted from https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/iddpm/__init__.py
"""

from functools import partial
import torch
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
        "--max_steps", type=int, default=1000, help="Max training steps"
    )
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
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
    return parser.parse_args()


def train_on_latents(
    device="cuda",
    batch_size=2,
    max_steps=1000,
    log_every=10,
    sample_every=100,
    depth=6,
    hidden_size=768,
    patch_size=(1, 2, 2),
    num_heads=3,
    diffusion_steps=1000,
    track=False,
    **kwargs,
):
    import numpy as np
    from cvlization.torch.net.vae.video_vqvae import VQVAE
    from stdit.model import STDiT

    # load from numpy
    token_ids = np.load("flying_mnist_tokens_32frames_train.npy")
    assert len(token_ids.shape) == 4, f"Expected 4D tensor, got {token_ids.shape}"
    # convert token_ids to embeddings
    vae = VQVAE.from_pretrained("zzsi_kungfu/videogpt/model-kbu39ped:v11")
    vae.eval()
    vae.to(device)
    token_ids = torch.tensor(token_ids.astype(np.int32), dtype=torch.long).to(device)

    with torch.no_grad():
        z = vae.vq.codes_to_vec(token_ids)
        assert len(z.shape) == 5, f"Expected 5D tensor, got {z.shape}"
        assert (
            z.shape[2] == token_ids.shape[1]
        ), f"Expected the temporal dimension has size {token_ids.shape[1]}, got {z.shape[2]}"
        # print(z.shape)  # (1000, 4, 8, 64, 64)

    def get_batch():
        idx = np.random.choice(len(z), batch_size, replace=False)
        return torch.Tensor(z[idx]).to(device)

    denoiser = STDiT(
        # depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs
        input_size=z.shape[2:],
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
        unconditional=True,
    ).to(device)
    diffusion = IDDPM(
        num_sampling_steps=diffusion_steps,
    )
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-4)

    # training loop
    for i in range(max_steps):
        x = get_batch()
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        loss_dict = diffusion.training_losses(
            model=denoiser, x_start=x, t=t, model_kwargs=None
        )

        # Backward & update
        loss = loss_dict["loss"].mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Callbacks

        if i % log_every == 0:
            print(f"Step {i}: {loss.item()}")
            if track:
                import wandb

                wandb.log({"train/loss": loss.item()})

        if i % sample_every == 0:
            with torch.no_grad():
                samples = diffusion.sample_unconditional(
                    model=denoiser,
                    # text_encoder=None,
                    n_samples=1,
                    z_size=z.shape[1:],
                    # prompts=[],  # ["a", "b"],
                    device=device,
                    additional_args=None,
                )
                # print(samples.shape)
                # TODO: multiply a scaler factor to latents to make mean = 0, std = 1
                # decode z into a video
                assert samples.shape[1:] == z.shape[1:], f"shape of samples is {samples.shape}, shape of z is {z.shape}"
                video = vae.decoder(samples)
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
