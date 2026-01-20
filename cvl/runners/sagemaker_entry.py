#!/usr/bin/env python3
"""
SageMaker entry point script for CVL examples.

This script is injected into the Docker container and handles:
1. Reading hyperparameters from SageMaker config
2. Setting up the environment
3. Running the training command
4. Copying outputs to SageMaker's model directory

SageMaker paths:
- /opt/ml/input/config/hyperparameters.json - training hyperparameters
- /opt/ml/input/data/<channel>/ - input data channels
- /opt/ml/model/ - output model artifacts (copied to S3)
- /opt/ml/output/ - output for failure messages
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_hyperparameters() -> dict:
    """Load hyperparameters from SageMaker config."""
    config_path = Path("/opt/ml/input/config/hyperparameters.json")
    if config_path.exists():
        with open(config_path) as f:
            # SageMaker wraps values in quotes, parse them
            raw = json.load(f)
            return {k: json.loads(v) if v.startswith('"') else v for k, v in raw.items()}
    return {}


def setup_environment():
    """Setup environment variables for training."""
    # Ensure unbuffered output for real-time logs
    os.environ["PYTHONUNBUFFERED"] = "1"

    # SageMaker provides these, but set defaults if missing
    os.environ.setdefault("SM_MODEL_DIR", "/opt/ml/model")
    os.environ.setdefault("SM_OUTPUT_DIR", "/opt/ml/output")

    # Set up data directory if present
    data_dir = Path("/opt/ml/input/data/training")
    if data_dir.exists():
        os.environ["SM_CHANNEL_TRAINING"] = str(data_dir)


def run_training(command: str, args: list) -> int:
    """Run the training command."""
    # Build full command
    full_cmd = command.split() + args

    print(f"Running: {' '.join(full_cmd)}")
    print("-" * 50)

    # Run training
    result = subprocess.run(full_cmd, cwd="/workspace")

    return result.returncode


def copy_outputs():
    """Copy outputs to SageMaker model directory."""
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Common output locations in CVL examples
    output_dirs = [
        Path("/workspace/outputs"),
        Path("/workspace/checkpoints"),
        Path("/workspace/logs"),
    ]

    for src_dir in output_dirs:
        if src_dir.exists():
            dst_dir = model_dir / src_dir.name
            print(f"Copying {src_dir} -> {dst_dir}")
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)


def write_failure(message: str):
    """Write failure message for SageMaker."""
    failure_path = Path("/opt/ml/output/failure")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(message)


def main():
    print("=" * 50)
    print("CVL SageMaker Entry Point")
    print("=" * 50)

    try:
        # Load hyperparameters
        hyperparams = load_hyperparameters()
        print(f"Hyperparameters: {hyperparams}")

        # Get training command (default: python train.py)
        command = hyperparams.get("command", "python train.py")
        args_json = hyperparams.get("args", "[]")
        args = json.loads(args_json) if isinstance(args_json, str) else args_json

        print(f"Command: {command}")
        print(f"Args: {args}")

        # Setup environment
        setup_environment()

        # Run training
        exit_code = run_training(command, args)

        if exit_code == 0:
            # Copy outputs to model directory
            copy_outputs()
            print("Training completed successfully")
        else:
            write_failure(f"Training failed with exit code {exit_code}")
            print(f"Training failed with exit code {exit_code}")

        sys.exit(exit_code)

    except Exception as e:
        error_msg = f"Entry point error: {e}"
        print(error_msg)
        write_failure(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
