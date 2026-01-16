#!/usr/bin/env python3
"""
Test script for SageMaker runner with nanogpt example.

Usage:
    AWS_PROFILE=nomad python test_sagemaker_runner.py

Before running, update the configuration below:
- ROLE_ARN: Your SageMaker execution role ARN
- OUTPUT_BUCKET: Your S3 bucket for training outputs
"""

import os

# Set AWS profile before importing boto3
os.environ.setdefault("AWS_PROFILE", "nomad")

# ============================================================
# CONFIGURATION - Update these values
# ============================================================

# SageMaker execution role ARN
# Must have permissions for: SageMaker, ECR, S3, CloudWatch
ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::595425361112:role/CVLSageMakerExecutionRole"
)

# S3 bucket for outputs (must exist)
OUTPUT_BUCKET = os.environ.get("SAGEMAKER_OUTPUT_BUCKET", "kfai-experimental-facture-cache")

# AWS region - must match S3 bucket region
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# Instance type
# Note: GPU instances require service quota increase
# Using CPU for testing (ml.m5.xlarge or ml.m5.large)
INSTANCE_TYPE = "ml.m5.xlarge"

# ============================================================


def main():
    # Validate configuration
    if "ACCOUNT_ID" in ROLE_ARN or "YOUR_SAGEMAKER_ROLE" in ROLE_ARN:
        print("ERROR: Please update ROLE_ARN in this script or set SAGEMAKER_ROLE_ARN env var")
        print(f"  Current value: {ROLE_ARN}")
        return 1

    if "your-bucket" in OUTPUT_BUCKET:
        print("ERROR: Please update OUTPUT_BUCKET in this script or set SAGEMAKER_OUTPUT_BUCKET env var")
        print(f"  Current value: {OUTPUT_BUCKET}")
        return 1

    # Import runner
    from cvl.runners import SageMakerRunner

    print("=" * 60)
    print("SageMaker Runner Test - nanogpt")
    print("=" * 60)
    print(f"AWS Profile: {os.environ.get('AWS_PROFILE', 'default')}")
    print(f"Region: {REGION}")
    print(f"Role ARN: {ROLE_ARN}")
    print(f"Output path: s3://{OUTPUT_BUCKET}/cvl-test/nanogpt/")
    print(f"Instance type: {INSTANCE_TYPE}")
    print("=" * 60)

    # Create runner
    runner = SageMakerRunner(
        role_arn=ROLE_ARN,
        region=REGION,
    )

    # Run training job
    # nanogpt needs config/train_shakespeare_char.py to specify the dataset
    exit_code = runner.run(
        example="nanogpt",
        preset="train",
        # Use CPU for testing (no GPU quota in this account)
        # Note: nanogpt's train.sh runs: python train.py config/train_shakespeare_char.py
        # Reduce eval_iters from 100 to 5 for faster CPU testing
        args=["config/train_shakespeare_char.py",
              "--max_iters=10", "--eval_interval=10", "--eval_iters=5",
              "--log_interval=5", "--device=cpu", "--compile=False"],
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{OUTPUT_BUCKET}/cvl-test/nanogpt/",

        # Quick test settings
        max_run_minutes=30,       # 30 min timeout
        spot=False,               # Use on-demand for faster start
        volume_size_gb=30,        # Minimal EBS
        download_outputs=True,    # Download results after training
    )

    print("=" * 60)
    if exit_code == 0:
        print("SUCCESS: SageMaker training completed")
    else:
        print(f"FAILED: Exit code {exit_code}")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
