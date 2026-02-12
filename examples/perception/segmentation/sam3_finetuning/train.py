#!/usr/bin/env python3
"""SAM3 Fine-tuning — main entry point."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch.multiprocessing

from dataset import (
    download_and_convert_hf_dataset,
    prepare_dataset_structure,
    verify_dataset_structure,
)

# Patch set_start_method so SAM3's internal call doesn't fail when
# the context was already set (e.g. by HF dataset loading).
_orig_set_start_method = torch.multiprocessing.set_start_method
torch.multiprocessing.set_start_method = lambda method, force=False: _orig_set_start_method(method, force=True)


# ---------------------------------------------------------------------------
# LR scheduler (injected into sam3.train.optim.schedulers at runtime)
# ---------------------------------------------------------------------------

class WarmupConstantParamScheduler:
    """Constant LR with linear warmup and cooldown."""

    def __init__(self, base_lr: float, warmup_steps: int = 100, cooldown_steps: int = 200):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps

    def __call__(self, step: int, where: float):
        lr = self.base_lr
        if self.warmup_steps and step < self.warmup_steps:
            lr = lr * step / self.warmup_steps
        if self.cooldown_steps and where > 0:
            total_steps = step / where
            remaining = total_steps - step
            if remaining < self.cooldown_steps:
                lr = lr * remaining / self.cooldown_steps
        return lr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM3 on COCO-format segmentation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to COCO dataset root (train/, valid/ with _annotations.coco.json)")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HuggingFace dataset to download")
    parser.add_argument("--output-dir", type=str, default="outputs/sam3_finetuning",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM3 checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="sam3-finetuning")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Monkey-patches applied before training starts
# ---------------------------------------------------------------------------

def _apply_patches():
    """Register custom scheduler and patch matcher for NaN safety."""
    import sam3.train.optim.schedulers as _sched_mod
    _sched_mod.WarmupConstantParamScheduler = WarmupConstantParamScheduler

    # Patch matcher to replace NaN/Inf in cost matrices with a large value
    # so linear_sum_assignment doesn't crash.
    from sam3.train import matcher as _matcher_mod
    _orig_do_matching = _matcher_mod._do_matching

    def _safe_do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
        if not np.all(np.isfinite(cost)):
            cost = np.copy(cost)
            cost[~np.isfinite(cost)] = 1e8
        return _orig_do_matching(cost, repeats=repeats, return_tgt_indices=return_tgt_indices, do_filtering=do_filtering)

    _matcher_mod._do_matching = _safe_do_matching


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_sam3_training(
    dataset_dir: str,
    output_dir: str,
    num_gpus: int,
    epochs: int = 20,
    wandb_args: dict | None = None,
):
    """Run SAM3 training using its native training infrastructure."""
    print("\n" + "=" * 80)
    print("STARTING SAM3 TRAINING")
    print("=" * 80)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    dataset_dir = prepare_dataset_structure(dataset_dir)

    try:
        from hydra import initialize_config_dir, compose
        from hydra.core.global_hydra import GlobalHydra
        from sam3.train.utils.train_utils import register_omegaconf_resolvers
        from sam3.train.train import single_node_runner, add_pythonpath_to_sys_path

        _apply_patches()

        GlobalHydra.instance().clear()
        config_dir = str(Path(__file__).parent / "configs")
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        register_omegaconf_resolvers()

        overrides = [
            f"paths.dataset_root={dataset_dir}",
            f"paths.experiment_log_dir={output_dir}",
            f"launcher.gpus_per_node={num_gpus}",
            f"trainer.max_epochs={epochs}",
        ]
        if wandb_args:
            overrides.append("trainer.logging.tensorboard_writer.should_log=False")
        cfg = compose(config_name="sam3_finetune", overrides=overrides)

        print("=" * 80)
        print("Training Configuration:")
        print(f"  Dataset: {cfg.paths.dataset_root}")
        print(f"  Output:  {cfg.paths.experiment_log_dir}")
        print(f"  GPUs:    {cfg.launcher.gpus_per_node}")
        print(f"  Epochs:  {cfg.trainer.max_epochs}")
        print("=" * 80 + "\n")

        add_pythonpath_to_sys_path()

        if wandb_args:
            from wandb_logging import setup_wandb_logging
            setup_wandb_logging(
                project=wandb_args["project"],
                run_name=wandb_args.get("run_name"),
                config={"epochs": epochs, "dataset": dataset_dir, "num_gpus": num_gpus},
                output_dir=output_dir,
                dataset_dir=dataset_dir,
            )

        main_port = random.randint(10000, 20000)
        single_node_runner(cfg, main_port)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_sam3_environment():
    """Ensure SAM3 is importable."""
    import subprocess

    sam3_source = Path("/opt/sam3")
    if not sam3_source.exists():
        print("ERROR: SAM3 source not found at /opt/sam3")
        sys.exit(1)

    sam3_path = str(sam3_source.absolute())
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    try:
        import sam3
        print(f"✓ SAM3 module loaded from: {sam3.__file__}")
    except ImportError as e:
        print(f"ERROR: Cannot import SAM3: {e}")
        print("Installing SAM3 package...")
        import shutil
        sam3_copy = Path("/tmp/sam3")
        if not sam3_copy.exists():
            shutil.copytree(sam3_source, sam3_copy, symlinks=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(sam3_copy)], check=True)
        print("✓ SAM3 installed successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 80)
    print("SAM3 Fine-tuning")
    print("=" * 80)

    # Step 1: Dataset
    if not args.dataset_dir and not args.hf_dataset:
        print("\n[1/4] Creating synthetic test shapes dataset...")
        from dataset_builder import DatasetBuilder
        DatasetBuilder()
        args.dataset_dir = "data/test_shapes"
    elif args.hf_dataset:
        print("\n[1/4] Downloading and converting HuggingFace dataset...")
        args.dataset_dir = download_and_convert_hf_dataset(args.hf_dataset)
    else:
        print(f"\n[1/4] Using provided dataset: {args.dataset_dir}")

    # Step 2: Verify
    print(f"\n[2/4] Verifying dataset structure...")
    if not verify_dataset_structure(args.dataset_dir):
        sys.exit(1)

    # Step 3: SAM3 environment
    print("\n[3/4] Setting up SAM3 environment...")
    setup_sam3_environment()

    # Step 4: Train
    print("\n[4/4] Running SAM3 training...")
    wandb_args = None
    if args.wandb:
        wandb_args = {"project": args.wandb_project, "run_name": args.wandb_run_name}

    run_sam3_training(
        dataset_dir=str(Path(args.dataset_dir).absolute()),
        output_dir=str(Path(args.output_dir).absolute()),
        num_gpus=args.num_gpus,
        epochs=args.epochs,
        wandb_args=wandb_args,
    )

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print(f"  Checkpoints: {args.output_dir}/checkpoints/")
    print("=" * 80)


if __name__ == "__main__":
    main()
