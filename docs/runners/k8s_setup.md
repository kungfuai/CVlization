# Kubernetes Runner Setup

Run CVL examples as Kubernetes Jobs on any cluster (EKS, GKE, AKS, local).

## Prerequisites

1. **kubectl configured** with cluster access
2. **kubernetes Python client**: `pip install kubernetes` or `pip install -e .[k8s]`

## Quick Start

```python
from cvl.runners import K8sRunner

runner = K8sRunner(namespace="default")

runner.run(
    image="my-registry/nanogpt:latest",
    command=["python", "train.py", "--max_iters=1000"],
    gpu=1,
    memory="8Gi",
)
```

## Cluster Setup

### Verify Access

```bash
# Check cluster connection
kubectl cluster-info

# Check available nodes
kubectl get nodes

# Check GPU nodes (if using NVIDIA)
kubectl get nodes -l nvidia.com/gpu.present=true
```

### Namespace Setup

```bash
# Create namespace for training jobs
kubectl create namespace ml-training

# Verify
kubectl get namespace ml-training
```

## GPU Configuration

### NVIDIA GPU Operator

For GPU support, your cluster needs the NVIDIA GPU Operator or device plugin:

```bash
# Check if GPU resources are available
kubectl describe nodes | grep nvidia.com/gpu
```

If not installed, see:
- EKS: https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
- GKE: GPU nodes have drivers pre-installed
- AKS: https://learn.microsoft.com/en-us/azure/aks/gpu-cluster

### Node Selectors

Target specific GPU nodes:

```python
runner.run(
    image="my-image",
    command=["python", "train.py"],
    gpu=1,
    node_selector={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB"},
)
```

Common node labels:
| Label | Example Values |
|-------|---------------|
| `nvidia.com/gpu.product` | `NVIDIA-A100-SXM4-40GB`, `Tesla-V100-SXM2-16GB` |
| `node.kubernetes.io/instance-type` | `p3.2xlarge`, `n1-standard-8` |
| `topology.kubernetes.io/zone` | `us-east-1a` |

## Private Container Registries

### Create Image Pull Secret

```bash
# For Docker Hub
kubectl create secret docker-registry regcred \
    --docker-server=https://index.docker.io/v1/ \
    --docker-username=<username> \
    --docker-password=<password> \
    -n ml-training

# For ECR (AWS)
kubectl create secret docker-registry ecr-cred \
    --docker-server=<account>.dkr.ecr.<region>.amazonaws.com \
    --docker-username=AWS \
    --docker-password=$(aws ecr get-login-password) \
    -n ml-training
```

### Use in Runner

```python
runner.run(
    image="my-private-registry/image:tag",
    command=["python", "train.py"],
    image_pull_secrets=["regcred"],
)
```

## Resource Limits

```python
runner.run(
    image="my-image",
    command=["python", "train.py"],
    gpu=1,              # Number of GPUs
    cpu="4",            # CPU cores
    memory="16Gi",      # Memory limit
    timeout_minutes=60, # Job timeout
)
```

## Troubleshooting

### "Pod pending" / Not Scheduled

```bash
# Check pod events
kubectl describe pod -l job-name=<job-name> -n ml-training

# Common causes:
# - Insufficient resources (GPU/memory)
# - Node selector doesn't match any node
# - Image pull failed
```

### "ImagePullBackOff"

```bash
# Check if secret exists
kubectl get secrets -n ml-training

# Check pod events for details
kubectl describe pod <pod-name> -n ml-training
```

### View Job Logs

```bash
# Get logs from job
kubectl logs -l job-name=<job-name> -n ml-training --follow
```

### Clean Up Stuck Jobs

```bash
# List jobs
kubectl get jobs -n ml-training

# Delete job
kubectl delete job <job-name> -n ml-training
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KUBECONFIG` | Path to kubeconfig file (default: ~/.kube/config) |
