# CVL Deploy

Deploy CVlization examples to serverless GPU platforms.

## Supported Platforms

- **Cerebrium** - Serverless GPU platform with auto-scaling

## Quick Start

```bash
# Deploy ltx2 video generation to Cerebrium
cvl deploy ltx2 --gpu L40

# Dry run (see what would be deployed without deploying)
cvl deploy ltx2 --gpu L40 --dry-run
```

## Cerebrium Deployment

### Prerequisites

1. **Install dependencies for model upload:**
   ```bash
   pip install huggingface_hub  # or: pip install cvl[deploy]
   ```

2. **Install Cerebrium CLI:**
   ```bash
   uv add cerebrium  # or: pip install cerebrium
   ```

3. **Login to Cerebrium:**
   ```bash
   cerebrium login
   ```

4. **Set up secrets** (for gated models like Gemma):
   - Go to [Cerebrium Dashboard](https://dashboard.cerebrium.ai)
   - Navigate to **Secrets** tab
   - Add `HF_TOKEN` with your HuggingFace token

   Secrets are automatically available as environment variables in deployed containers.

### GPU Options

| GPU | Identifier | VRAM | Plan |
|-----|------------|------|------|
| T4 | TURING_T4 | 16GB | Hobby+ |
| L4 | ADA_L4 | 24GB | Hobby+ |
| A10 | AMPERE_A10 | 24GB | Hobby+ |
| L40 | ADA_L40 | 48GB | Hobby+ |
| A100 | AMPERE_A100_40GB | 40GB | Enterprise |
| A100_80GB | AMPERE_A100_80GB | 80GB | Enterprise |
| H100 | HOPPER_H100 | 80GB | Enterprise |

```bash
# Use specific GPU
cvl deploy ltx2 --gpu L40
cvl deploy ltx2 --gpu A100
```

### Project ID

If you have multiple Cerebrium projects, specify the project ID:

```bash
cvl deploy ltx2 --gpu L40 --project p-xxxxxxxx
```

## Model Upload

During deployment, `cvl deploy` automatically uploads required models to Cerebrium's persistent storage. This eliminates slow cold starts caused by downloading large models at runtime.

### Models Uploaded by Default

**ltx2:** (~50GB total)
| Model | Files | Size | Description |
|-------|-------|------|-------------|
| `Lightricks/LTX-2` | `ltx-2-19b-distilled-fp8.safetensors` | ~26GB | FP8 distilled checkpoint |
| `Lightricks/LTX-2` | `ltx-2-spatial-upscaler-x2-1.0.safetensors` | ~1GB | Spatial upscaler |
| `google/gemma-3-12b-it-qat-q4_0-unquantized` | (full repo) | ~23GB | Text encoder (gated, requires HF token) |

Note: Only the specific files needed for inference are uploaded, not the full 190GB LTX-2 repo.

### How It Works

1. Checks if model exists in Cerebrium persistent storage
2. If not, downloads to local HuggingFace cache (if not already cached)
3. Uploads to Cerebrium via `cerebrium cp`
4. Deployed container uses `HF_HOME=/persistent-storage/.cache/huggingface`

### Skip Model Upload

To skip model upload (if models are already uploaded or you want to handle manually):

```bash
cvl deploy ltx2 --gpu L40 --skip-model-upload
```

### Manual Model Upload

If you need additional models not included by default:

```bash
# Download model locally first
python -c "from huggingface_hub import snapshot_download; snapshot_download('org/model-name')"

# Upload to Cerebrium
cerebrium cp ~/.cache/huggingface/hub/models--org--model-name .cache/huggingface/hub/models--org--model-name
```

## How It Works

`cvl deploy` generates a deployment package with:

1. **cerebrium.toml** - Configuration for hardware, scaling, and custom runtime
2. **Dockerfile** - Custom Docker image with FastAPI server
3. **main.py** - FastAPI endpoints (`/health`, `/ready`, `/run`)

The custom Dockerfile approach gives full control over the environment and avoids issues with Cerebrium's default runtime.

### Generated Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/run` | POST | Run inference |

### Example Request

```bash
curl -X POST https://api.aws.us-east-1.cerebrium.ai/v4/{project_id}/ltx2/run \
  -H "Authorization: Bearer $CEREBRIUM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walking on a sunny beach",
    "height": 512,
    "width": 768,
    "num_frames": 33
  }'
```

### Response Format

```json
{
  "video_base64": "AAAAIGZ0eXBpc29t...",
  "format": "mp4",
  "width": 768,
  "height": 512,
  "num_frames": 33,
  "frame_rate": 24.0,
  "mode": "t2v"
}
```

## Managing Deployed Services

```bash
# List deployed services
cvl services list

# Check service status
cvl services status ltx2

# View logs
cvl services logs ltx2

# Invoke service
cvl services invoke ltx2 --prompt "A dog running"

# Delete service
cvl services delete ltx2
```

## Local Testing

Before deploying, you can test the generated Docker image locally:

```bash
# Generate deployment files without deploying
cvl deploy ltx2 --gpu L40 --dry-run

# Build and run locally
cd /tmp/cvl-deploy-ltx2-xxxxx
docker build -t ltx2-test .
docker run -d --gpus all -p 8192:8192 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ltx2-test

# Test endpoints
curl http://localhost:8192/health
curl http://localhost:8192/ready
```

## Supported Examples

Currently supported for automatic deployment:

- **ltx2** - LTX-2 video generation (T2V and I2V)

See [TODO.md](./TODO.md) for plans to support more examples.

## Troubleshooting

### "Cannot access gated repo" error

Add your `HF_TOKEN` to Cerebrium secrets:
1. Go to Cerebrium Dashboard → Secrets
2. Add `HF_TOKEN` with your HuggingFace token

### Deployment fails to start

Check logs:
```bash
cvl services logs ltx2
```

### Timeout on first request

If models are uploaded to persistent storage, first request loads models (~60s).
If models are NOT uploaded, first request downloads them (can take 30+ minutes for large models).

Use `cvl deploy` without `--skip-model-upload` to ensure models are pre-uploaded.

## Future Improvements

### Webhook Support for Long Tasks (TODO)

Cerebrium has a 3-minute timeout per request. For long-running tasks (high-resolution video, 100+ frames), add webhook support:

```python
def run(prompt, webhook_endpoint=None):
    if webhook_endpoint:
        # Return immediately, POST result to webhook later
        run_id = str(uuid.uuid4())
        background_task(generate_and_post, prompt, webhook_endpoint, run_id)
        return {"status": "processing", "run_id": run_id}
    else:
        # Synchronous (must complete in <3 min)
        return generate(prompt)
```

Current ltx2 (33 frames, 512x768) completes in ~30-60 seconds, so webhook is not urgent.

### Tensorizer Support (TODO)

[Tensorizer](https://github.com/coreweave/tensorizer) can speed up model loading by 30-50% for large models (20B+ parameters). It converts models to protocol buffer format for faster GPU loading.

To add Tensorizer support:
1. Serialize models with `TensorSerializer` during upload
2. Deserialize with `TensorDeserializer` during inference
3. Add `tensorizer>=2.7.0` to dependencies

See [Cerebrium docs on faster cold starts](https://docs.cerebrium.ai/cerebrium/other-topics/faster-cold-starts) for details.
