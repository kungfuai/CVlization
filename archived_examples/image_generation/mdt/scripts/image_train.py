"""
Train a diffusion model on images.
"""

import argparse
import torch

from masked_diffusion import dist_util, logger
from masked_diffusion.image_datasets import load_data
from masked_diffusion.resample import create_named_schedule_sampler
from masked_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from masked_diffusion.train_util import TrainLoop
from masked_diffusion import create_diffusion, model_and_diffusion_defaults, diffusion_defaults
import masked_diffusion.models as models_mdt

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist_multinode(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(configs)
    print(args)
    image_size = configs['image_size']
    latent_size = image_size // 8
    model = models_mdt.__dict__[args.model](input_size=latent_size, mask_ratio=args.mask_ratio, decode_layer=args.decode_layer)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f} M")
    diffusion = create_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.latents_input_file is not None:
        from cvlization.dataset.flying_mnist import FlyingMNISTImageLatentsBuilder

        db = FlyingMNISTImageLatentsBuilder(args.latents_input_file)
        train_ds = db.training_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True)
        def gen():
            while True:
                yield from train_loader
        data = gen()
    else:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

    logger.log("training...")
    if args.latents_input_file is not None:
        print("Using latents input file:", args.latents_input_file)
    
    if args.track:
        import wandb

        wandb.init(project=args.project)
        wandb.config.update(args)
        # wandb.watch(model)
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        data_is_latent=(args.latents_input_file is not None),
        track=args.track,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model="MDTv2_S_2",
        mask_ratio=None,
        decode_layer=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument(
        "--rank", default=0, type=int, help="""rank for distrbuted training."""
    )
    parser.add_argument(
        "--latents_input_file",
        type=str,
        default="data/latents/flying_mnist_11k__sd-vae-ft-mse_latents_32frames_train.npy",
    )
    parser.add_argument(
        "--track",
        action="store_true",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="mdt",
        help="Weights and biases project",
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
