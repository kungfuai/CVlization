from argparse import ArgumentParser, Namespace
from latte import Latte_models
import torch
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
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--clip_max_norm", type=float, default=0.5)
    parser.add_argument("--start_clip_iter", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    args = parser.parse_args()
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
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
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
            x = rearrange(x, 'b c f h w -> b f c h w')
            # video_name = video_data['video_name']
            # x = x.to(device)
            # y = y.to(device) # y is text prompt; no need put in gpu
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                # print("x:", x.shape)
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
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
                # write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                # write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            
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