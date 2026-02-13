# SkyPilot Runner Setup

Run CVL examples on any cloud (AWS, GCP, Azure, Lambda Labs) with automatic cost optimization.

## Prerequisites

1. **SkyPilot installed** with cloud provider support
2. **Cloud credentials configured**

## Installation

```bash
# For AWS
pip install "skypilot[aws]"

# For GCP
pip install "skypilot[gcp]"

# For Azure
pip install "skypilot[azure]"

# For multiple clouds
pip install "skypilot[aws,gcp]"

# All clouds
pip install "skypilot[all]"
```

## Cloud Credential Setup

### AWS

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Verify
sky check aws
```

### GCP

```bash
# Install gcloud CLI, then:
gcloud auth login
gcloud auth application-default login

# Verify
sky check gcp
```

### Azure

```bash
# Install Azure CLI, then:
az login

# Verify
sky check azure
```

### Lambda Labs

```bash
# Set API key
export LAMBDA_API_KEY=your_key

# Verify
sky check lambda
```

## Quick Start

```python
from cvl.runners import SkyPilotRunner

runner = SkyPilotRunner()

runner.run(
    command="python train.py --max_iters=1000",
    workdir="./examples/nanogpt",
    gpu="A100:1",
)
```

## GPU Types

Common GPU specifications:

| Spec | Description |
|------|-------------|
| `A100:1` | 1x NVIDIA A100 |
| `A100:8` | 8x NVIDIA A100 |
| `V100:4` | 4x NVIDIA V100 |
| `T4:1` | 1x NVIDIA T4 (cheapest) |
| `A10G:1` | 1x NVIDIA A10G |
| `L4:1` | 1x NVIDIA L4 |

## Spot Instances

Save up to 70% with spot/preemptible instances:

```python
runner.run(
    command="python train.py",
    workdir="./examples/nanogpt",
    gpu="A100:1",
    use_spot=True,  # Enable spot instances
)
```

SkyPilot handles:
- Automatic checkpointing
- Spot interruption recovery
- Failover to on-demand if needed

## Cloud Selection

```python
# Specific cloud
runner.run(
    command="python train.py",
    gpu="A100:1",
    cloud="aws",        # or "gcp", "azure", "lambda"
    region="us-east-1", # optional
)

# Let SkyPilot choose cheapest
runner.run(
    command="python train.py",
    gpu="A100:1",
    # No cloud specified = auto-select cheapest
)
```

## File Syncing

SkyPilot automatically syncs your `workdir` to the remote instance:

```python
runner.run(
    command="python train.py",
    workdir="./my-project",  # Synced to remote
)
```

For additional files:

```python
runner.run(
    command="python train.py",
    workdir="./my-project",
    file_mounts={
        "/data": "./local-data",  # Mount local dir
        "/models": "s3://bucket/models",  # Mount S3
    },
)
```

## Setup Commands

Run setup before your main command:

```python
runner.run(
    command="python train.py",
    workdir="./my-project",
    setup="pip install -r requirements.txt",
)
```

## Cost Optimization Tips

1. **Use spot instances** for fault-tolerant training
2. **Let SkyPilot choose the cloud** - it finds the cheapest option
3. **Set auto-stop** to avoid idle costs (default: 10 min)
4. **Use smaller GPUs for debugging** (T4, L4)

Check current prices:
```bash
sky show-gpus --all
```

## Troubleshooting

### "No cloud access"

```bash
# Check which clouds are configured
sky check

# Re-authenticate
aws configure  # or gcloud auth login, az login
```

### "No resources available"

Try:
1. Different region: `region="us-west-2"`
2. Different GPU: `gpu="V100:1"` instead of `A100`
3. Different cloud: `cloud="gcp"` instead of `aws`
4. Enable spot: `use_spot=True`

### View Running Clusters

```bash
# List clusters
sky status

# SSH into cluster
sky ssh <cluster-name>

# View logs
sky logs <cluster-name>
```

### Clean Up

```bash
# Stop cluster (can restart later)
sky stop <cluster-name>

# Terminate cluster
sky down <cluster-name>

# Terminate all
sky down -a
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account JSON |
| `LAMBDA_API_KEY` | Lambda Labs API key |
