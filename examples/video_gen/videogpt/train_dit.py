from argparse import ArgumentParser, Namespace
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
        extras=args.extras
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
    diffusion_steps=1000
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
        loss_type=loss_type
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
        reconstructions = rearrange(reconstructions, "b t c h w -> b c t h w")

    if reconstructions.device.type != "cpu":
        reconstructions = reconstructions.cpu()

    # make sure the pixel values are in the range [0, 1]
    reconstructions = (reconstructions - reconstructions.min()) / (
        reconstructions.max() - reconstructions.min() + 1e-6
    )
    reconstructions = (reconstructions * 255).to(torch.uint8)

    b = min(reconstructions.shape[0], 1)
    panel_name = "sample"

    # side by side video
    reconstructions = rearrange(reconstructions, "b c t h w -> c t h (b w)")
    display = reconstructions

    display = wandb.Video(data_or_path=display, fps=4, format="mp4")
    wandb.log({f"{panel_name}/samples": display})

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Latte-S/2")
    parser.add_argument("--sequence_length", type=int, default=16, help="This can be number of frames or number of video segments where each segment has multiple frames.")
    parser.add_argument("--extras", type=int, default=1)  # 1: unconditional
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--learn_sigma", action="store_true")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_dit_model(args)
    model = model.to(device)
    print(model)
    diffusion = create_diffusion(
        timestep_respacing=""
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
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
        resolution=args.resolution, max_frames_per_video=args.sequence_length
    )
    train_ds = dataset_builder.training_dataset()
    val_ds = dataset_builder.validation_dataset()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

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
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     continue

            x = video_data['video'].to(device, non_blocking=True)
            assert len(x.shape) == 5
            assert x.shape[1] == 3
            # print("x:", x.shape)
            # x: torch.Size([2, 3, 4, 256, 256])            
            x = rearrange(x, 'b c f h w -> b f c h w')
            # video_name = video_data['video_name']
            # x = x.to(device)
            # y = y.to(device) # y is text prompt; no need put in gpu
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                # print("encoder input x:", x.shape)
                # encoder input x: torch.Size([8, 3, 256, 256])
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # print("encoded x:", x.shape, x.dtype)
                # encoded x: torch.Size([8, 4, 32, 32])
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                video_name = video_data['video_name']
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)

            # x_flattened = rearrange(x, 'b f c h w -> (b f) c h w')
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=True)

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
                print(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
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
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                ae_temporal_stride = 1
                ae_space_stride = 8
                resolution = args.resolution
                latent_size = [int(resolution / ae_space_stride), int(resolution / ae_space_stride)]
                print("in_channels:", model.in_channels)
                z = torch.randn(1, int(args.sequence_length), model.in_channels, latent_size[0], latent_size[1], device=device)
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
                print("sampling...")
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
                # print("samples:", samples.shape)
                # samples: torch.Size([1, 4, 4, 32, 32])
                # Decode samples:
                samples = rearrange(samples, 'b f c h w -> b c f h w')
                flattened_samples = rearrange(samples, 'b c f h w -> (b f) c h w')
                decoder_output = vae.decode(flattened_samples)
                decoded_flattened_samples = decoder_output.sample
                # print("decoded_flattened_samples:", decoded_flattened_samples.shape)
                decoded_samples = rearrange(decoded_flattened_samples, '(b f) c h w -> b f c h w', b=1)
                print("decoded_samples:", decoded_samples.shape, torch.float32)
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
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()