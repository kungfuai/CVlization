# CVL Remote Runners

Execute CVL examples on remote machines with automatic setup and cleanup.

## Features

- **SSH Runner**: Run on any remote host via SSH (requires CVL installed on remote)
- **Docker Context Runner**: Rsync files + SSH execution (no remote CVL needed)
- **Lambda Labs Runner**: Automated instance lifecycle management
- **SageMaker Runner**: AWS managed training with spot instances and S3 artifacts
- **Automatic cleanup**: Guaranteed instance termination on exit/error/Ctrl+C
- **Timeout support**: Kill runaway jobs with remote shutdown
- **Cost estimation**: See pricing before launching
- **Progress streaming**: Real-time output from remote jobs

## Installation

```bash
# Install CVL with SSH remote execution
pip install -e .[remote]

# Install CVL with AWS SageMaker support
pip install -e .[aws]

# Install all remote execution options
pip install -e .[remote,aws]
```

## Usage

### SSH Runner (Any Remote Host)

Use this for existing instances (Lambda, AWS, your own server, etc.)

```python
from cvl.runners import SSHRunner

runner = SSHRunner()

runner.run_remote(
    host="ubuntu@123.45.67.89",
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000"],
    timeout_minutes=120,  # Optional: kill after 2 hours
    setup_cvl=True  # Auto-install CVL if needed
)
```

**From command line:**
```bash
# Coming soon: cvl run --ssh command
python -m cvl.runners.example_usage ssh
```

### Lambda Labs Runner (Automated)

Automatically creates instance, runs job, and terminates.

```python
from cvl.runners import LambdaLabsRunner

# Set API key
export LAMBDA_API_KEY=your_key_here

runner = LambdaLabsRunner()

runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000", "--batch_size=16"],
    gpu_type="gpu_1x_a100_sxm4",
    timeout_minutes=120
)
```

**From command line:**
```bash
# Coming soon: cvl run --remote lambda command
python -m cvl.runners.example_usage lambda
```

### Docker Context Runner (Rsync + SSH)

Best for "I have a GPU VM somewhere" - syncs files via rsync, runs on remote.

```python
from cvl.runners import DockerContextRunner

runner = DockerContextRunner()

runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000"],
    host="ubuntu@gpu-server",  # SSH host or config alias

    # Optional
    remote_workdir="/tmp/cvl-remote",  # Where to sync files
    sync_outputs=True,                  # Rsync outputs back
    timeout_minutes=120,
)
```

**How it works:**
1. Rsyncs example directory to remote machine
2. SSHs to remote and runs the script there
3. Docker commands run on remote (where files exist)
4. Rsyncs outputs back to local machine

**Difference from SSHRunner:**
- Uses rsync for efficient delta file transfer
- Doesn't require CVL to be installed on remote
- Better for iterative development (fast re-sync)

### SageMaker Runner (AWS Managed Training)

Best for production ML training with artifact management and spot instances.

```python
from cvl.runners import SageMakerRunner

runner = SageMakerRunner(
    role_arn="arn:aws:iam::123456789:role/SageMakerExecutionRole",
    region="us-east-1"  # Optional, defaults to AWS_DEFAULT_REGION
)

runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=1000"],
    instance_type="ml.g5.xlarge",
    output_path="s3://my-bucket/training-outputs/",

    # Optional parameters
    spot=True,                    # Use spot instances (up to 70% cheaper)
    max_wait_minutes=60,          # Max wait for spot capacity
    max_run_minutes=120,          # Job timeout
    entry_command="python train.py",  # Override default command
    download_outputs=True,        # Download artifacts after training
)
```

**Prerequisites:**
1. AWS credentials configured (`aws configure` or environment variables)
2. IAM role with SageMaker, ECR, S3, and CloudWatch permissions
3. Docker installed locally (for building images)
4. S3 bucket for outputs

**How it works:**
1. Builds Docker image from example's Dockerfile
2. Pushes to ECR (creates repository if needed)
3. Starts SageMaker training job
4. Streams CloudWatch logs in real-time
5. Downloads outputs from S3 on completion

### Kubernetes Runner (Any K8s Cluster)

Run jobs on any Kubernetes cluster (EKS, GKE, AKS, local).

```python
from cvl.runners import K8sRunner

runner = K8sRunner(namespace="ml-training")

runner.run(
    image="my-registry/nanogpt:latest",
    command=["python", "train.py", "--max_iters=1000"],
    gpu=1,
    memory="8Gi",
    timeout_minutes=60,
)
```

**Prerequisites:**
1. Valid kubeconfig (`~/.kube/config`) or in-cluster config
2. Container image accessible from cluster
3. `pip install kubernetes`

**How it works:**
1. Creates Kubernetes Job with specified resources
2. Waits for pod to be scheduled
3. Streams logs in real-time
4. Cleans up job on completion/interrupt

## Configuration

### SSH Authentication

The SSH runner uses your default SSH key (`~/.ssh/id_rsa`). To use a different key:

```python
runner = SSHRunner(ssh_key_path="/path/to/key")
```

### Lambda Labs API Key

Set your API key via environment variable:

```bash
export LAMBDA_API_KEY=your_api_key_here
```

Or pass it directly:

```python
runner = LambdaLabsRunner(api_key="your_key")
```

## Timeout Behavior

Both runners support timeouts:

```python
runner.run_remote(
    ...,
    timeout_minutes=120,  # Kill job after 2 hours
    timeout_action="sudo shutdown -h now"  # What to do on timeout
)
```

**How it works:**
- Remote command runs with `timeout 120m cvl run ...`
- If timeout occurs (exit code 124), runs `timeout_action`
- For Lambda Labs, instance shuts down → billing stops
- Local cleanup always runs regardless

**Exit codes:**
- `0`: Success
- `1-123`: Command error (local cleanup runs)
- `124`: Timeout (remote action runs, then local cleanup)
- `130`: Ctrl+C (local cleanup runs)

## Cost Safety

Users are responsible for monitoring costs. The runner will terminate instances on completion, error, or timeout, but check your Lambda Labs dashboard for any orphaned instances.

## Cleanup Guarantees

**Lambda Labs Runner guarantees cleanup via:**
1. ✅ Try/finally block (normal case)
2. ✅ atexit handler (Python exit)
3. ✅ Signal handlers (Ctrl+C, SIGTERM)
4. ✅ Remote timeout action (runaway jobs)

**Only fails if:**
- ❌ Kill -9 (can't catch)
- ❌ Power outage

For those cases, manually terminate via Lambda dashboard or API.

## Examples

### Train GPT-2 on A100
```python
runner = LambdaLabsRunner()
runner.run(
    example="nanogpt",
    preset="train",
    args=["--max_iters=5000", "--batch_size=32"],
    gpu_type="gpu_1x_a100_sxm4",
    timeout_minutes=180  # 3 hour safety net
)
```

### Run Inference on Existing Instance
```python
runner = SSHRunner()
runner.run_remote(
    host="ubuntu@my-lambda.lambdalabs.com",
    example="mixtral8x7b",
    preset="predict",
    args=[],
    setup_cvl=False  # Already installed
)
```

### Quick Test Run
```python
runner = LambdaLabsRunner()
runner.run("nanogpt", "train", ["--max_iters=100"], timeout_minutes=10)
```

## Architecture

```
Runner                 Workflow                          Backend
─────────────────────────────────────────────────────────────────────
SSHRunner              ssh → run → stream               paramiko
LambdaLabsRunner       create → ssh → run → terminate   Lambda API
DockerContextRunner    rsync → ssh → run → rsync back   rsync + SSH
SageMakerRunner        build → push → train → download  AWS APIs
K8sRunner              create job → stream → cleanup    Kubernetes API
```

**Lambda Labs Runner:**
- Instance creation via API
- Wait for SSH ready
- Delegates execution to SSH runner
- Handles cleanup

**SSH Runner:**
- Works with any SSH host
- Auto-installs CVL if needed
- Proper argument escaping (prevents shell injection)
- Timeout with remote actions
- Real-time output streaming

**Docker Context Runner:**
- Rsyncs files to remote (efficient delta transfer)
- Doesn't require CVL installed on remote
- Better for iterative development
- Syncs outputs back automatically

**SageMaker Runner:**
- Builds and pushes Docker image to ECR
- Creates managed training job
- Streams CloudWatch logs
- Downloads artifacts from S3
- Spot instance support for cost savings

**K8s Runner:**
- Creates Kubernetes Job with specified resources
- Streams pod logs in real-time
- Automatic cleanup on completion/interrupt
- Works with any k8s cluster (EKS, GKE, AKS, local)

## Troubleshooting

**"paramiko not installed"**
```bash
pip install paramiko
```

**"LAMBDA_API_KEY not set"**
```bash
export LAMBDA_API_KEY=your_key
```

**"Authentication failed"**
- Check your SSH key: `ssh-add -l`
- Add key to remote: `ssh-copy-id ubuntu@host`

**"Instance not ready after 300s"**
- Lambda may be slow or out of capacity
- Try different region or GPU type

**"Instance not terminated"**
- Check Lambda dashboard
- Manually terminate if needed
- Check for orphaned instances regularly

**"boto3 not installed"**
```bash
pip install boto3
# Or: pip install -e .[aws]
```

**"ECR push failed"**
- Check AWS credentials: `aws sts get-caller-identity`
- Ensure IAM permissions for ECR (ecr:GetAuthorizationToken, ecr:CreateRepository, etc.)

**"SageMaker training job failed"**
- Check CloudWatch logs in AWS Console
- Verify IAM role has S3 and ECR access
- Check S3 bucket exists and is accessible

## Future Enhancements

- [ ] CLI integration: `cvl run --ssh host example preset`
- [ ] CLI integration: `cvl run --remote lambda example preset`
- [ ] CLI integration: `cvl run --sagemaker example preset`
- [ ] State file for orphaned instance cleanup
- [ ] EC2 Runner (thin wrapper around SSHRunner with EC2 lifecycle)
- [ ] Modal Runner (serverless GPU)
- [ ] RunPod Runner (similar to Lambda Labs)
- [ ] GCP Vertex AI Runner
- [ ] Heartbeat pattern for bulletproof cleanup
- [ ] Multi-instance support (distributed training)
- [x] ~~Artifact download (save checkpoints locally)~~ - Implemented in SageMaker runner

## Development

Run tests:
```bash
pytest cvl/tests/test_runner.py
```

See `example_usage.py` for more examples.
