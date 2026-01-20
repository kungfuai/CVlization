# SkyPilot Runner Design

Design notes and TODOs for the SkyPilot runner.

## Overview

SkyPilot runner provides multi-cloud execution for CVL examples via [SkyPilot](https://skypilot.readthedocs.io/). It abstracts cloud providers (AWS, GCP, Azure, Lambda Labs) and handles instance provisioning, file syncing, and cost optimization.

## Current Status

**Implemented:**
- Basic runner class (`cvl/runners/skypilot_runner.py`)
- CLI integration (`cvl run --runner skypilot`)
- GPU specification (`A100:1`, `V100:4`, etc.)
- Cloud/region selection
- Spot instance support
- Workdir syncing
- Auto-stop and teardown
- Cleanup handlers (Ctrl+C, exit, signals)

**Not implemented:**
- Docker container support (runs directly on VM)
- Model upload to persistent storage
- Example-aware deployment (like `cvl deploy`)
- Managed storage volumes

## Future: Cloud-Specific Runners

We may implement cloud-specific runner flags that use SkyPilot under the hood:

```bash
# Current (explicit SkyPilot)
cvl run example train --runner skypilot --cloud aws --gpu A100:1

# Future (cloud as runner, SkyPilot hidden)
cvl run example train --runner ec2 --gpu A100:1
cvl run example train --runner gcp --gpu A100:1
cvl run example train --runner azure --gpu A100:1
cvl run example train --runner lambda --gpu A100:1
```

### Rationale

1. **Simpler UX** - Users think in terms of clouds, not orchestration tools
2. **Implementation detail hidden** - SkyPilot is the engine, not the interface
3. **Consistent naming** - `--runner ec2` parallels `--runner sagemaker`

### Implementation

Each cloud runner would be a thin wrapper around SkyPilot:

```python
# In __main__.py
if runner in ("ec2", "gcp", "azure", "lambda"):
    cloud_map = {"ec2": "aws", "gcp": "gcp", "azure": "azure", "lambda": "lambda"}
    return _run_skypilot(..., cloud=cloud_map[runner])
```

### AWS Services Used (for --runner ec2)

SkyPilot uses these AWS services:

| Service | Purpose |
|---------|---------|
| EC2 | Launch/manage GPU instances |
| IAM | Instance role (`skypilot-v1`) |
| S3 | Optional: storage mounts |
| VPC | Default or configured VPC |

**Not used:** ECS, EKS, SageMaker, Lambda, Batch

Required IAM permissions: `ec2:RunInstances`, `ec2:TerminateInstances`, `ec2:Describe*`, `iam:PassRole`, etc.
See [SkyPilot AWS Permissions](https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/aws.html).

### Comparison: --runner ec2 vs --runner sagemaker

| Aspect | ec2 (via SkyPilot) | sagemaker |
|--------|-------------------|-----------|
| Compute | Raw EC2 instances | Managed containers |
| Permissions | EC2 + IAM | SageMaker + ECR + S3 + CloudWatch |
| Docker | Not used | Required (builds & pushes image) |
| Artifacts | Manual | Auto-uploaded to S3 |
| Spot recovery | Basic | Built-in checkpointing |
| Cost | EC2 pricing | +20-30% SageMaker markup |
| Best for | Simple runs, multi-cloud | Production training, managed infra |

## Architecture

```
User                    SkyPilot Runner                 Cloud
─────────────────────────────────────────────────────────────────
cvl run example    →    Build sky.Task           →    Provision VM
    --runner skypilot   Set resources (GPU, cloud)     Install deps
    --gpu A100:1        sky.launch()             →    Sync workdir
                        stream_logs=True         ←    Stream output
                        sky.down()               →    Terminate VM
```

### Key Components

1. **`SkyPilotRunner` class** - Main entry point
   - `run()` - Low-level: run arbitrary command
   - `run_example()` - High-level: run CVL example directory

2. **Task configuration** - Maps to `sky.Task`
   - `command` - What to run
   - `workdir` - Directory to sync
   - `setup` - Pre-run setup commands

3. **Resource configuration** - Maps to `sky.Resources`
   - `gpu` - Accelerator spec
   - `cloud` - Provider preference
   - `region` - Location preference
   - `use_spot` - Spot/preemptible instances

## Comparison with Other Runners

| Feature | SkyPilot | Cerebrium | SageMaker | K8s |
|---------|----------|-----------|-----------|-----|
| Multi-cloud | Yes | No | No | Yes* |
| Spot instances | Yes | No | Yes | Yes |
| Docker support | No** | Yes | Yes | Yes |
| Persistent storage | No | Yes | S3 | PVC |
| Serverless | No | Yes | No | No |
| Auto-scaling | No | Yes | No | Yes |
| Cost optimization | Yes | Manual | Manual | Manual |

*K8s requires cluster per cloud
**SkyPilot runs on VM, not in container

## Design Decisions

### Why no Docker support?

SkyPilot provisions VMs and runs commands directly. This is simpler but has tradeoffs:

**Pros:**
- No Docker build step
- Full VM access (can install anything)
- Easy debugging (SSH in)

**Cons:**
- No reproducible environment guarantee
- Setup commands run every launch
- Slower cold starts

**Future:** Consider SkyPilot's [Docker support](https://skypilot.readthedocs.io/en/latest/examples/docker.html) for containerized execution.

### Why workdir sync instead of persistent storage?

SkyPilot syncs local `workdir` to remote on each launch. This works for code but not for large models.

**Current limitation:** Large models must be downloaded on each VM launch (slow cold starts).

**Future:** Integrate with cloud storage (S3, GCS) or SkyPilot's [Storage](https://skypilot.readthedocs.io/en/latest/reference/storage.html) for persistent model caching.

## TODOs

### High Priority

- [ ] **Docker support** - Use SkyPilot's Docker execution mode
  ```python
  task.set_resources(sky.Resources(
      image_id="docker:my-image:latest"
  ))
  ```

- [ ] **Persistent model storage** - Upload models to cloud storage once
  ```python
  runner.upload_models(
      repo_id="Lightricks/LTX-2",
      files=["ltx-2-19b-distilled-fp8.safetensors"],
      storage="s3://my-bucket/models/"
  )
  ```

- [ ] **Example-aware deployment** - Auto-detect requirements from example.yaml
  ```bash
  cvl deploy example --runner skypilot --cloud aws
  ```

### Medium Priority

- [ ] **Managed volumes** - Use SkyPilot Storage for HuggingFace cache
  ```python
  task.set_storage_mounts({
      "/models": sky.Storage(name="hf-cache", source="s3://models/")
  })
  ```

- [ ] **Multi-node training** - Distributed training across VMs
  ```python
  runner.run(
      command="torchrun --nproc_per_node=8 train.py",
      gpu="A100:8",
      num_nodes=2,  # NEW
  )
  ```

- [ ] **Job queuing** - Queue multiple jobs with dependencies
  ```python
  runner.queue([
      {"command": "python preprocess.py"},
      {"command": "python train.py", "depends_on": 0},
  ])
  ```

- [ ] **Cost estimation** - Show estimated cost before launch
  ```bash
  cvl run example --runner skypilot --gpu A100:1 --estimate-cost
  # Estimated cost: $2.50/hr (AWS us-east-1 spot)
  ```

### Low Priority

- [ ] **Checkpoint sync** - Periodically sync checkpoints to local/S3
- [ ] **Spot recovery** - Better handling of spot interruptions
- [ ] **Cluster reuse** - Keep cluster alive for iterative development
- [ ] **Resource auto-detection** - Read GPU requirements from example.yaml
- [ ] **Serve mode** - Deploy as endpoint (like `cvl deploy` for Cerebrium)

## Integration with cvl deploy

Currently `cvl deploy` only supports Cerebrium. Future integration with SkyPilot:

```bash
# Current (Cerebrium only)
cvl deploy ltx2 --gpu L40

# Future (SkyPilot)
cvl deploy ltx2 --platform skypilot --cloud aws --gpu A100
```

### What this would do:

1. **Build Docker image** from example Dockerfile
2. **Push to registry** (ECR for AWS, GCR for GCP)
3. **Upload models** to cloud storage (S3/GCS)
4. **Generate SkyPilot YAML** with:
   - Docker image reference
   - Storage mounts for models
   - GPU/cloud configuration
5. **Deploy as service** using `sky serve`

### SkyPilot Serve

SkyPilot has a [serve](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) feature for deploying endpoints:

```yaml
# service.yaml
service:
  readiness_probe: /health
  replicas: 2

resources:
  accelerators: A100:1

run: |
  python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

```bash
sky serve up service.yaml
```

This could be the foundation for `cvl deploy --platform skypilot`.

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service account |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription |
| `LAMBDA_API_KEY` | Lambda Labs API key |

### Example YAML Override

For advanced configuration, users can provide a SkyPilot YAML:

```python
runner.run_from_yaml("my-task.yaml")
```

## Testing

```bash
# Unit tests (mocked)
pytest cvl/tests/test_skypilot_runner.py

# Integration test (requires cloud credentials)
python -c "
from cvl.runners import SkyPilotRunner
runner = SkyPilotRunner()
runner.run(
    command='echo hello',
    gpu='T4:1',
    use_spot=True,
    down=True,
)
"
```

## References

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [SkyPilot GitHub](https://github.com/skypilot-org/skypilot)
- [SkyPilot Docker Support](https://skypilot.readthedocs.io/en/latest/examples/docker.html)
- [SkyPilot Storage](https://skypilot.readthedocs.io/en/latest/reference/storage.html)
- [SkyPilot Serve](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html)
