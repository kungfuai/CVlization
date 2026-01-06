#!/usr/bin/env python3
"""
Run nanogpt training on SageMaker.

This script demonstrates running the nanogpt example on AWS SageMaker using the
SageMakerRunner. The training uses the Shakespeare character-level dataset.

Environment variables (from setup):
- AWS_PROFILE: default
- AWS_DEFAULT_REGION: us-east-1
- SAGEMAKER_ROLE_ARN: arn:aws:iam::595425361112:role/CVLSageMakerExecutionRole
- SAGEMAKER_OUTPUT_BUCKET: cvl-training-outputs-1767711732
"""

import os
from cvl.runners import SageMakerRunner

def main():
    # Get configuration from environment or use defaults
    role_arn = os.getenv(
        "SAGEMAKER_ROLE_ARN",
        "arn:aws:iam::595425361112:role/CVLSageMakerExecutionRole"
    )
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    output_bucket = os.getenv(
        "SAGEMAKER_OUTPUT_BUCKET",
        "cvl-training-outputs-1767711732"
    )

    # Initialize SageMaker runner
    runner = SageMakerRunner(
        role_arn=role_arn,
        region=region
    )

    # Run nanogpt training
    print("=" * 60)
    print("Starting nanogpt training on SageMaker")
    print("=" * 60)
    print(f"Role ARN: {role_arn}")
    print(f"Region: {region}")
    print(f"Output bucket: {output_bucket}")
    print(f"Instance type: ml.g5.xlarge")
    print(f"Dataset: Shakespeare character-level")
    print(f"Max iterations: 5000")
    print("=" * 60)

    runner.run(
        example="nanogpt",
        preset="train",
        args=["config/train_shakespeare_char.py", "--max_iters=5000"],
        instance_type="ml.g5.xlarge",
        output_path=f"s3://{output_bucket}/nanogpt-training/",
        spot=False,  # Change to True for 70% cost savings (but may be interrupted)
        max_run_minutes=60,
        download_outputs=True,  # Download artifacts after training
    )

    print("\n" + "=" * 60)
    print("Training job completed!")
    print("=" * 60)
    print(f"Outputs downloaded to: ./training-outputs/")
    print("Check CloudWatch logs for detailed training metrics")

if __name__ == "__main__":
    main()
