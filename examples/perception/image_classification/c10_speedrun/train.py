#!/usr/bin/env python
import argparse
import logging
import time

import numpy as np
import torch

if not hasattr(np, "float"):
    np.float = float  # Ensure davidnet utilities remain compatible with numpy>=2.0

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CIFAR10 speed-run pipelines (HyperLightBench or DavidNet)."
    )
    parser.add_argument(
        "--pipeline",
        choices=("hlb", "davidnet"),
        default="hlb",
        help="Pipeline to execute.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target torch.device (e.g. cuda, mps, cpu, or auto). Defaults to cuda.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override the number of training epochs (float for HLB, int for DavidNet).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the DavidNet training batch size (ignored for HLB).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run_davidnet(device: torch.device, epochs: int, batch_size: int) -> dict:
    from cvlization.torch.net.image_classification.davidnet.dawn_utils import (
        Network,
        net,
    )
    from cvlization.torch.trainer.david_trainer import DavidTrainer

    LOGGER.info(
        "Running DavidNet on %s for %s epochs (batch size %s)",
        device,
        epochs,
        batch_size,
    )

    model = Network(net()).to(device)
    if device.type == "cuda":
        model = model.half()
    else:
        model = model.float()
    trainer = DavidTrainer(
        model=model,
        epochs=epochs,
        train_batch_size=batch_size,
        use_cached_cifar10=True,
        device=device,
        train_dataset=None,
        val_dataset=None,
    )

    sync_device(device)
    start = time.time()
    trainer.train()
    sync_device(device)
    elapsed = time.time() - start

    LOGGER.info("DavidNet run finished in %.2f seconds", elapsed)
    return {"elapsed_seconds": elapsed}


def run_hlb(device: torch.device, train_epochs: float) -> dict:
    from cvlization.torch.training_pipeline.image_classification import hlb

    if device.type != "cuda":
        raise ValueError("HLB pipeline currently requires a CUDA device.")

    LOGGER.info("Running HyperLightBench for %.2f epochs", train_epochs)
    original_epochs = hlb.hyp["misc"]["train_epochs"]
    hlb.hyp["misc"]["train_epochs"] = float(train_epochs)
    try:
        sync_device(device)
        start = time.time()
        ema_val_acc = hlb.main()
        sync_device(device)
        elapsed = time.time() - start
    finally:
        hlb.hyp["misc"]["train_epochs"] = original_epochs

    ema_val = float(ema_val_acc) if ema_val_acc is not None else None
    LOGGER.info(
        "HyperLightBench run finished in %.2f seconds (EMA val acc: %s)",
        elapsed,
        f"{ema_val:.4f}" if ema_val is not None else "n/a",
    )
    return {"elapsed_seconds": elapsed, "ema_val_acc": ema_val}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    device = resolve_device(args.device)

    if args.pipeline == "davidnet":
        epochs = int(args.epochs) if args.epochs is not None else 12
        batch_size = int(args.batch_size) if args.batch_size is not None else 512
        results = run_davidnet(device=device, epochs=epochs, batch_size=batch_size)
        print(
            f"RESULT pipeline=davidnet device={device.type} elapsed_seconds={results['elapsed_seconds']:.2f} "
            f"epochs={epochs} batch_size={batch_size}"
        )
    else:
        train_epochs = float(args.epochs) if args.epochs is not None else 11.5
        results = run_hlb(device=device, train_epochs=train_epochs)
        summary = (
            f" ema_val_acc={results['ema_val_acc']:.4f}"
            if results["ema_val_acc"] is not None
            else ""
        )
        print(
            f"RESULT pipeline=hlb device={device.type} elapsed_seconds={results['elapsed_seconds']:.2f} "
            f"train_epochs={train_epochs}{summary}"
        )


if __name__ == "__main__":
    main()
