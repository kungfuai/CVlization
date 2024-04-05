from argparse import ArgumentParser, Namespace
import os
from latte import Latte_models
import torch
import wandb
from time import time
from torch import nn
from einops import rearrange
import gaussian_diffusion as gd
from respace import SpacedDiffusion, space_timesteps
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL
from latte import clip_grad_norm_


def create_dit_model(args: Namespace) -> nn.Module:
    model = Latte_models[args.model](
        input_size=args.latent_input_size,
        num_classes=args.num_classes,
        num_frames=args.sequence_length,
        learn_sigma=args.learn_sigma,
        extras=args.extras,
    )
    return model


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    # learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
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


def log_samples(reconstructions):
    """
    Log the reconstructions to wandb.

    :param reconstructions: (B, T, C, H, W) reconstructed videos
    """
    import wandb

    # make sure the shape is corect
    if reconstructions.shape[2] != 3:
        reconstructions = rearrange(reconstructions, "b c t h w -> b t c h w")

    if reconstructions.device.type != "cpu":
        reconstructions = reconstructions.cpu()

    # make sure the pixel values are in the range [0, 1]
    reconstructions = (reconstructions - reconstructions.min()) / (
        reconstructions.max() - reconstructions.min() + 1e-6
    )
    reconstructions = (reconstructions * 255).to(torch.uint8)

    b = min(reconstructions.shape[0], 1)
    panel_name = "sample"

    reconstructions = rearrange(reconstructions, "b t c h w -> t c h (b w)")
    # print("video:", reconstructions.shape)  # video: torch.Size([16, 3, 256, 256])
    display = reconstructions

    display = wandb.Video(data_or_path=display, fps=4, format="mp4")
    wandb.log({f"{panel_name}/samples": display})


def load_model_from_wandb(
    model_full_name: str = "zzsi_kungfu/videogpt/model-tjzu02pg:v17",
) -> dict:
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


def create_vae(
    wandb_model_name: str = None, hf_model_name: str = "stabilityai/sd-vae-ft-mse"
) -> AutoencoderKL:
    if wandb_model_name:
        vae = load_model_from_wandb(wandb_model_name)
        return vae
    vae = AutoencoderKL.from_pretrained(hf_model_name)
    return vae


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Latte-S/2")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=16,
        help="This is the latent sequence length. If ae_temporal_stride is 1, then it is the same as the number of video frames.",
    )
    parser.add_argument("--extras", type=int, default=1)  # 1: unconditional
    parser.add_argument("--num_clips_per_video", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--ae_spatial_stride", type=int, default=8)
    parser.add_argument("--ae_temporal_stride", type=int, default=1)
    parser.add_argument("--latent_input_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--clip_max_norm", type=float, default=0.5)
    parser.add_argument("--start_clip_iter", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--track", action="store_true")

    args = parser.parse_args()
    if args.track:
        wandb.init(project="flying_mnist")
        wandb.config.update(args)

    # Do this for 3090
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_dit_model(args)
    model = model.to(device)
    print(model)
    diffusion = create_diffusion(timestep_respacing="")
    if ":" in args.vae_model:
        # it is a wandb model name
        print("Loading VAE from wandb...")
        vae = create_vae(wandb_model_name=args.vae_model)
    else:
        print("Loading VAE from Hugging Face...")
        vae = AutoencoderKL.from_pretrained(args.vae_model)

    vae.eval()
    vae = vae.to(device)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    first_epoch = 0
    num_train_epochs = args.epochs

    # TODO: hardcode dataset
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder

    dataset_builder = FlyingMNISTDatasetBuilder(
        # TODO: this is assuming the VAE is image-based and not video-based
        # If it is video-based VAE, then max_frames_per_video should be sequence_length * vae_temporal_compression_factor
        resolution=args.resolution,
        max_frames_per_video=100,
        # max_frames_per_video=args.sequence_length * args.ae_temporal_stride
    )
    train_ds = dataset_builder.training_dataset()
    val_ds = dataset_builder.validation_dataset()
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    start_time = time()
    for epoch in range(first_epoch, num_train_epochs):
        # sampler.set_epoch(epoch)
        for _, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     continue

            # TODO: get multiple clips for each long video
            num_clips_per_video = args.num_clips_per_video
            long_video = video_data["video"].to(device, non_blocking=True)
            for _ in range(num_clips_per_video):
                num_frames = args.sequence_length * args.ae_temporal_stride
                random_start = torch.randint(0, long_video.shape[2] - num_frames, (1,))
                x = long_video[:, :, random_start : random_start + num_frames, :, :]
                # x = video_data['video'].to(device, non_blocking=True)
                assert len(x.shape) == 5
                assert x.shape[1] == 3
                # print("x:", x.shape)
                # x: torch.Size([2, 3, 4, 256, 256])
                x = rearrange(x, "b c f h w -> b f c h w")
                # video_name = video_data['video_name']
                # x = x.to(device)
                # y = y.to(device) # y is text prompt; no need put in gpu
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    b, _, _, _, _ = x.shape
                    # print("encoder input x:", x.shape)
                    # encoder input x: torch.Size([8, 3, 256, 256])
                    if isinstance(vae, AutoencoderKL):
                        x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    else:
                        assert hasattr(vae, "encoder")
                        # print("input video:", x.shape)  # input video: torch.Size([2, 16, 3, 256, 256])
                        x = rearrange(x, "b f c h w -> b c f h w").contiguous()
                        encoded = vae.encoder(x)
                        if isinstance(encoded, dict):
                            z = encoded["z"]
                            # mu = encoded["mu"]
                            # logvar = encoded["logvar"]
                        else:
                            z = encoded
                        vq_output = vae.vq(z)
                        if isinstance(vq_output, dict):
                            z = vq_output["z_recon"]
                        else:
                            z = vq_output
                        z = rearrange(z, "b d f h w -> (b f) d h w")
                        # x = z
                        x_unscaled = z
                        latent_multiplier = 9
                        latent_bias = -0.27
                        x = (
                            z * latent_multiplier + latent_bias
                        )  # TODO: this is specific to the VAE! Should be automatically estimated from the VAE.

                        x_unscaled = rearrange(
                            x_unscaled, "(b f) d h w -> b f d h w", b=b
                        ).contiguous()

                    # print(f"latents: mean={x.mean()}, std={x.std()}, min={x.min()}, max={x.max()}")
                    x = rearrange(x, "(b f) d h w -> b f d h w", b=b).contiguous()
                if args.extras == 78:  # text-to-video
                    raise "T2V training are Not supported at this moment!"
                elif args.extras == 2:
                    video_name = video_data["video_name"]
                    model_kwargs = dict(y=video_name)
                else:
                    model_kwargs = dict(y=None)

                # x_flattened = rearrange(x, 'b f c h w -> (b f) c h w')
                t = torch.randint(
                    0, diffusion.num_timesteps, (x.shape[0],), device=device
                )
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss.backward()

                if (
                    train_steps < args.start_clip_iter
                ):  # if train_steps >= start_clip_iter, will clip gradient
                    gradient_norm = clip_grad_norm_(
                        model.parameters(), args.clip_max_norm, clip_grad=False
                    )
                else:
                    gradient_norm = clip_grad_norm_(
                        model.parameters(), args.clip_max_norm, clip_grad=True
                    )

                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    # avg_loss = avg_loss.item() / dist.get_world_size()
                    # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                    #  mean=0.025898655876517296, std=0.9987635016441345, min=-10.309021949768066, max=7.061285495758057
                    print(
                        f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                    if args.track:
                        wandb.log({"train/loss": avg_loss.item()})
                        wandb.log({"train/gradient_norm": gradient_norm})

                    # write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                    # write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Generate samples:
                if (
                    args.sample_every > 0
                    and train_steps % args.sample_every == 0
                    and train_steps > 0
                ):
                    ae_space_stride = args.ae_spatial_stride
                    resolution = args.resolution
                    # TODO: latent_size can be inferred as resolution / ae_space_stride
                    # TODO: make sequence_length the video frame count, and calculate the latent sequence length with ae_temporal_stride
                    latent_size = [
                        int(resolution / ae_space_stride),
                        int(resolution / ae_space_stride),
                    ]
                    # print("in_channels:", model.in_channels)
                    z = torch.randn(
                        1,
                        int(args.sequence_length),
                        model.in_channels,
                        latent_size[0],
                        latent_size[1],
                        device=device,
                    )
                    using_cfg = False
                    if using_cfg:
                        z = torch.cat([z, z], 0)
                        y = torch.randint(0, args.num_classes, (1,), device=device)
                        y_null = torch.tensor([args.num_classes] * 1, device=device)
                        y = torch.cat([y, y_null], dim=0)
                        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                        sample_fn = model.forward_with_cfg
                    else:
                        sample_fn = model.forward
                        model_kwargs = dict(y=None)
                    # print("sampling... z:", z.shape)  # z: torch.Size([1, 4, 4, 32, 32])
                    samples = diffusion.p_sample_loop(
                        sample_fn,
                        z.shape,
                        z,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=True,
                        device=device,
                    )
                    # print("samples:", samples.shape)
                    # samples: torch.Size([1, 4, 4, 32, 32])
                    # Decode samples:
                    samples = rearrange(samples, "b f c h w -> b c f h w")
                    if isinstance(vae, AutoencoderKL):
                        flattened_samples = rearrange(
                            samples, "b c f h w -> (b f) c h w"
                        )
                        decoder_output = vae.decode(flattened_samples)
                        decoded_flattened_samples = decoder_output.sample
                        # print("decoded_flattened_samples:", decoded_flattened_samples.shape)
                        decoded_samples = rearrange(
                            decoded_flattened_samples, "(b f) c h w -> b f c h w", b=1
                        )
                    else:
                        assert hasattr(vae, "decoder")
                        decoded_samples = vae.decoder(
                            (samples - latent_bias) / latent_multiplier
                        )
                        decoded_samples = rearrange(
                            decoded_samples, "b c f h w -> b f c h w", b=1
                        )
                        # # Also decode a latent from training data.
                        # assert decoded_samples.shape[2] == 3
                        # x is the latent: shape is (b f d h w)
                        training_z = x_unscaled[:1, :, :, :, :]
                        training_z = rearrange(
                            training_z, "b f c h w -> b c f h w", b=1
                        )
                        # print("training_z:", training_z.shape)  # training_z: torch.Size([1, 4, 4, 64, 64])
                        training_latent_decoded = vae.decoder(training_z)
                        training_latent_decoded = rearrange(
                            training_latent_decoded, "b c f h w -> b f c h w", b=1
                        )
                        # side by side
                        # print("decoded_samples:", decoded_samples.shape)
                        # print("training_latent_decoded:", training_latent_decoded.shape)
                        decoded_samples = torch.cat(
                            [decoded_samples, training_latent_decoded], 4
                        )

                    # print("decoded_samples:", decoded_samples.shape, decoded_samples.dtype)

                    if args.track:
                        log_samples(decoded_samples)
                    # decoded_samples: torch.Size([1, 4, 3, 256, 256])

                # Save DiT checkpoint:
                checkpoint_dir = "checkpoints/dit"
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    # if rank == 0:
                    if True:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            # "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")


def extract_token_ids():
    from cvlization.dataset.flying_mnist import FlyingMNISTDatasetBuilder
    from latents import extract_token_ids
    from tqdm import tqdm
    import numpy as np

    max_frames_per_video = 32
    db = FlyingMNISTDatasetBuilder(
        max_frames_per_video=max_frames_per_video, resolution=256
    )
    train_ds = db.training_dataset()
    vae = create_vae(wandb_model_name="zzsi_kungfu/videogpt/model-kbu39ped:v11")
    vae = vae.to("cuda")
    all_token_ids = []
    for j, token_ids in tqdm(
        enumerate(extract_token_ids(vae, train_ds, batch_size=8, output_device="cpu"))
    ):
        # print(token_ids.shape, token_ids.dtype)
        all_token_ids.append(token_ids.numpy().reshape(1, -1))
        # print("all_token_ids:", all_token_ids[-1].astype(float).mean())
        # if j > 1:
        #     break
    all_token_ids = np.concatenate(all_token_ids, 0)
    print(all_token_ids.shape, all_token_ids.dtype)
    print(all_token_ids)
    # save
    np.save(
        f"flying_mnist_tokens_{max_frames_per_video}frames_train.npy", all_token_ids
    )


if __name__ == "__main__":
    # m = load_model_from_wandb()
    # import sys
    # sys.exit(0)

    main()
