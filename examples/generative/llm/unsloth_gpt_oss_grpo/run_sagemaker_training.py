#!/usr/bin/env python3
"""
Run GRPO training on SageMaker.

This script demonstrates running the unsloth_gpt_oss_grpo example on AWS SageMaker using the
SageMakerRunner. The training uses Group Relative Policy Optimization (GRPO) to fine-tune
GPT-OSS 20B for code generation tasks.

Required environment variables:
- SAGEMAKER_ROLE_ARN: IAM role ARN for SageMaker execution
- SAGEMAKER_OUTPUT_BUCKET: S3 bucket for training outputs

Optional environment variables:
- AWS_PROFILE: AWS CLI profile (default: default)
- AWS_DEFAULT_REGION: AWS region (default: us-east-1)
"""

import os
from cvl.runners import SageMakerRunner

def main():
    # Get configuration from environment variables
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    output_bucket = os.getenv("SAGEMAKER_OUTPUT_BUCKET")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Validate required environment variables
    if not role_arn:
        raise ValueError("SAGEMAKER_ROLE_ARN environment variable is required")
    if not output_bucket:
        raise ValueError("SAGEMAKER_OUTPUT_BUCKET environment variable is required")

    # Initialize SageMaker runner
    runner = SageMakerRunner(
        role_arn=role_arn,
        region=region
    )

    # Run GRPO training
    print("=" * 60)
    print("Starting GRPO training on SageMaker")
    print("=" * 60)
    print(f"Role ARN: {role_arn}")
    print(f"Region: {region}")
    print(f"Output bucket: {output_bucket}")
    print(f"Instance type: ml.g5.2xlarge (24GB VRAM)")
    print(f"Model: GPT-OSS 20B")
    print(f"Task: Code generation (matrix multiplication)")
    print(f"Max steps: 10")
    print("=" * 60)

    runner.run(
        example="unsloth-gpt-oss-grpo",
        preset="train",
        args=[],  # train.py uses config.yaml, no CLI args
        instance_type="ml.g5.2xlarge",  # 24GB VRAM, meets GRPO 15GB+ requirement
        output_path=f"s3://{output_bucket}/grpo-training/",
        spot=False,  # Change to True for 70% cost savings (but may be interrupted)
        max_run_minutes=120,  # GRPO takes ~15min for 10 steps, allow 2 hours
        download_outputs=True,  # Download artifacts after training
    )

    print("\n" + "=" * 60)
    print("Training job completed!")
    print("=" * 60)
    print(f"Outputs downloaded to: ./outputs/")
    print("Check CloudWatch logs for detailed training metrics")

if __name__ == "__main__":
    main()
