#!/bin/bash
# Setup SageMaker execution role for CVL training
#
# Usage:
#   AWS_PROFILE=nomad ./setup_sagemaker_role.sh

set -e

ROLE_NAME="CVLSageMakerExecutionRole"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_DEFAULT_REGION:-us-east-1}

echo "Creating SageMaker execution role: $ROLE_NAME"
echo "Account: $ACCOUNT_ID"
echo "Region: $REGION"

# Trust policy for SageMaker
TRUST_POLICY='{
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
}'

# Create the role
aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --description "Execution role for CVL SageMaker training jobs" \
    2>/dev/null || echo "Role may already exist, continuing..."

# Attach required policies
echo "Attaching policies..."

# SageMaker full access (for training)
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
    2>/dev/null || true

# ECR full access (for pushing/pulling images)
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess \
    2>/dev/null || true

# S3 full access (for outputs - can be restricted in production)
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
    2>/dev/null || true

# CloudWatch logs (for streaming logs)
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess \
    2>/dev/null || true

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo ""
echo "=========================================="
echo "SageMaker role created successfully!"
echo "=========================================="
echo "Role ARN: $ROLE_ARN"
echo ""
echo "To use this role, set the environment variable:"
echo "  export SAGEMAKER_ROLE_ARN=\"$ROLE_ARN\""
echo ""
echo "Or run the test script with:"
echo "  SAGEMAKER_ROLE_ARN=\"$ROLE_ARN\" AWS_PROFILE=nomad python test_sagemaker_runner.py"
