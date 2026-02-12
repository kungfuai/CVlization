#!/usr/bin/env python3
"""SAM3 LoRA fine-tuning — CLI entry point.

Wraps SAM3's Hydra-based trainer with a user-friendly argparse interface.
Loads the ``config_lora.yaml`` config and injects CLI overrides.

Example usage (inside Docker via train.sh)::

    python train_lora.py \\
        --dataset-dir /data/aquarium-combined \\
        --epochs 20 --lora-rank 64
"""

import logging
import os
import random
import sys
from argparse import ArgumentParser

import torch
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf
from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers

log = logging.getLogger(__name__)


def single_proc_run(local_rank, main_port, cfg, world_size, use_wandb=False):
    """Launch a single training process (mirrors sam3.train.train)."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass
    trainer = instantiate(cfg.trainer, _recursive_=False)

    if use_wandb and local_rank == 0:
        from wandb_logging import patch_trainer_for_wandb

        dataset_dir = OmegaConf.select(cfg, "paths.dataset_root")
        output_dir = OmegaConf.select(cfg, "paths.experiment_log_dir")
        patch_trainer_for_wandb(trainer, dataset_dir, output_dir, cfg.wandb)

    trainer.run()


def main():
    parser = ArgumentParser(
        description="SAM3 LoRA Fine-tuning",
        formatter_class=lambda prog: __import__(
            "argparse"
        ).ArgumentDefaultsHelpFormatter(prog),
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="COCO dataset root containing train/ and test/ subdirs",
    )
    parser.add_argument("--output-dir", default="/workspace/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--unfreeze-decoder", action="store_true")
    parser.add_argument("--unfreeze-seg-head", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--val-freq", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="sam3-lora", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    args = parser.parse_args()

    register_omegaconf_resolvers()

    # Load the self-contained LoRA config from the workspace directory.
    # We use OmegaConf.load() instead of Hydra compose because the config
    # lives in the workspace, not inside the installed sam3 package.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = OmegaConf.load(os.path.join(script_dir, "config_lora.yaml"))

    # Apply CLI overrides
    cli = OmegaConf.create(
        {
            "paths": {
                "dataset_root": args.dataset_dir,
                "experiment_log_dir": args.output_dir,
            },
            "trainer": {
                "max_epochs": args.epochs,
                "val_epoch_freq": args.val_freq,
            },
            "scratch": {
                "lora_rank": args.lora_rank,
                "lr_lora": args.lr,
                "unfreeze_decoder": args.unfreeze_decoder,
                "unfreeze_seg_head": args.unfreeze_seg_head,
            },
            "launcher": {
                "gpus_per_node": args.num_gpus,
            },
        }
    )
    cfg = OmegaConf.merge(cfg, cli)

    print("=" * 70)
    print("SAM3 LoRA Fine-tuning")
    print("=" * 70)
    print(f"  Dataset:  {args.dataset_dir}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LR:       {args.lr}")
    print(f"  GPUs:     {args.num_gpus}")
    if args.unfreeze_decoder:
        print("  Decoder:  UNFROZEN")
    if args.unfreeze_seg_head:
        print("  Seg head: UNFROZEN")
    print("=" * 70)

    makedir(args.output_dir)
    with g_pathmgr.open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Weights & Biases — init before trainer so metrics are logged cleanly
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dataset_dir": args.dataset_dir,
                "epochs": args.epochs,
                "lora_rank": args.lora_rank,
                "lr": args.lr,
                "unfreeze_decoder": args.unfreeze_decoder,
                "unfreeze_seg_head": args.unfreeze_seg_head,
                "num_gpus": args.num_gpus,
            },
        )
        # Use epoch as x-axis so metrics + images at the same epoch align.
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    cfg.launcher.num_nodes = 1
    main_port = random.randint(10000, 65000)

    if args.num_gpus == 1:
        single_proc_run(0, main_port, cfg, 1, use_wandb=args.wandb)
    else:
        torch.multiprocessing.set_start_method("spawn")
        torch.multiprocessing.start_processes(
            single_proc_run,
            args=(main_port, cfg, args.num_gpus, args.wandb),
            nprocs=args.num_gpus,
            start_method="spawn",
        )


if __name__ == "__main__":
    main()
