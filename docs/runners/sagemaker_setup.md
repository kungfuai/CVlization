# SageMaker Runner AWS Setup

This document describes the AWS resources and permissions required to run CVL examples on SageMaker.

## Prerequisites

1. **AWS CLI configured** with credentials that have admin-like permissions
2. **Docker installed locally** for building container images
3. **boto3 installed**: `pip install boto3` or `pip install -e .[aws]`

## Required AWS Resources

### 1. IAM Execution Role

SageMaker training jobs require an IAM role with permissions to:
- Pull images from ECR
- Read/write to S3
- Write CloudWatch logs

**Create the role using our setup script:**

```bash
AWS_PROFILE=your_profile ./docs/runners/setup_sagemaker_role.sh
```

Or manually create a role with these policies attached:
- `AmazonSageMakerFullAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonS3FullAccess`
- `CloudWatchLogsFullAccess`

The role must have a trust policy allowing SageMaker to assume it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 2. S3 Bucket

You need an S3 bucket to store training outputs. The SageMaker role must have read/write access to this bucket.

**Important:** The S3 bucket must be in the **same region** as your SageMaker training job. Cross-region access will fail with "authorization header is malformed; the region is wrong".

```bash
# Create bucket in specific region
aws s3 mb s3://your-bucket-name --region us-east-2
```

To check an existing bucket's region:
```bash
aws s3api get-bucket-location --bucket your-bucket-name
```

### 3. ECR Repository (Auto-created)

The SageMaker runner automatically creates ECR repositories for container images. Your AWS user/role needs these ECR permissions:
- `ecr:GetAuthorizationToken`
- `ecr:CreateRepository`
- `ecr:DescribeRepositories`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`
- `ecr:BatchGetImage`
- `ecr:GetDownloadUrlForLayer`

### 4. Service Quotas

By default, new AWS accounts have **zero quota** for SageMaker GPU instances. Check and request increases:

```bash
# List current quotas
aws service-quotas list-service-quotas --service-code sagemaker

# Request increase (example for ml.g4dn.xlarge)
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-678C1C90 \
    --desired-value 1
```

Common instance types and their quota codes:
| Instance Type | Quota Code | Use Case |
|--------------|------------|----------|
| ml.g4dn.xlarge | L-678C1C90 | Cheapest GPU, good for testing |
| ml.g5.xlarge | L-41F0D6B5 | Better GPU, A10G |
| ml.p3.2xlarge | L-8AA36237 | V100 GPU, production training |
| ml.m5.xlarge | L-BAC29C4D | CPU only, for testing |

## Quick Start

1. **Setup AWS resources:**
   ```bash
   # Set your profile
   export AWS_PROFILE=your_profile

   # Create SageMaker role
   ./docs/runners/setup_sagemaker_role.sh

   # Create S3 bucket (if needed)
   aws s3 mb s3://my-training-outputs
   ```

2. **Run a training job:**
   ```python
   from cvl.runners import SageMakerRunner

   runner = SageMakerRunner(
       role_arn="arn:aws:iam::123456789:role/CVLSageMakerExecutionRole",
       region="us-east-1"
   )

   runner.run(
       example="nanogpt",
       preset="train",
       args=["--max_iters=1000"],
       instance_type="ml.g4dn.xlarge",
       output_path="s3://my-training-outputs/nanogpt/",
   )
   ```

## Cost Considerations

- **On-demand pricing**: Full price, immediate availability
- **Spot instances**: Up to 70% cheaper, but can be interrupted
  ```python
  runner.run(..., spot=True, max_wait_minutes=30)
  ```

- **Terminate failed jobs**: The runner automatically stops jobs on error/interrupt

## Time Expectations

SageMaker jobs have overhead beyond training time:

| Phase | Typical Duration |
|-------|-----------------|
| Docker build & ECR push | 1-3 min (first run), seconds (cached) |
| Instance provisioning | 1-2 min |
| Docker image download | 1-2 min |
| Training | Varies |
| Artifact upload to S3 | 10-30 sec |

For quick iteration, test locally first:
```bash
# Build and test Docker image locally before submitting to SageMaker
docker run --rm your-image python train.py --max_iters=5
```

## Troubleshooting

### "ResourceLimitExceeded" Error
Your account has no quota for the requested instance type. Request a quota increase via AWS Service Quotas console.

### "S3 region mismatch" / "authorization header is malformed"
The S3 bucket is in a different region than SageMaker. Either:
1. Create a new bucket in the same region as SageMaker
2. Change the SageMaker region to match your bucket: `SageMakerRunner(region="us-east-2")`

### "AccessDenied" Error
Check that:
1. Your AWS credentials have ECR and SageMaker permissions
2. The SageMaker role ARN is correct
3. The S3 bucket is accessible by the SageMaker role

### "ImagePushFailed" Error
Check ECR permissions and that Docker is running locally.

### Training Job Fails Immediately
Check CloudWatch logs in the AWS Console under `/aws/sagemaker/TrainingJobs/{job-name}`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_PROFILE` | AWS CLI profile to use |
| `AWS_DEFAULT_REGION` | AWS region (default: us-east-1) |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN |
| `SAGEMAKER_OUTPUT_BUCKET` | S3 bucket for outputs |
