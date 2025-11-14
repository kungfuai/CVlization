# EGSTalker Dockerfile Optimizations

## Summary

Two build options are available, both optimized for A10 GPU:

| Build Type | Command | Image Size | Build Time | Use Case |
|------------|---------|------------|------------|----------|
| **Single-stage** | `bash build.sh` | ~20GB | ~15-18 min | Simple, development |
| **Multi-stage** | `bash build_multistage.sh` | ~10.5GB | ~15-18 min | Production, deployment |

## GPU-Specific Optimization

**A10 GPU** (compute capability 8.6)
- `TORCH_CUDA_ARCH_LIST="8.6"` (previously: "7.0;7.5;8.0;8.6;8.9;9.0")
- **Build time savings**: ~4-5 minutes (66% faster compilation)
- Compiles only for your specific GPU architecture

## Single-Stage Build (Dockerfile)

**Current approach**: Everything in one image

### Pros
- Simpler Dockerfile
- Good for development (has build tools for debugging)
- Easy to understand and modify

### Cons
- Larger image size (~20GB)
- Includes unnecessary build tools in final image
- Uses devel base (17.3GB) instead of runtime (7.7GB)

### What's included
- Base: pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel (17.3GB)
- Build tools: gcc, g++, build-essential (retained)
- All Python packages and compiled extensions
- EGSTalker source code

## Multi-Stage Build (Dockerfile.multistage)

**Optimized approach**: Separate build and runtime stages

### Pros
- **50% smaller** image (~10.5GB vs ~20GB)
- No build tools in final image (more secure)
- Uses runtime base (7.7GB) instead of devel (17.3GB)
- Same performance at runtime

### Cons
- Slightly more complex Dockerfile
- Can't compile additional extensions without rebuilding

### How it works
1. **Stage 1 (builder)**: Uses devel base, compiles everything
2. **Stage 2 (runtime)**: Uses runtime base, copies only:
   - `/opt/conda` (all pip packages)
   - `/workspace/egstalker` (source + compiled .so files)

### Space savings breakdown
```
Base image:       17.3GB → 7.7GB  (-9.6GB)
Build tools:      ~200MB → 0MB    (-200MB)
Total savings:    ~9.8GB
```

## Build Time Comparison

### Before Optimization
- CUDA architectures: 6 (7.0, 7.5, 8.0, 8.6, 8.9, 9.0)
- Gaussian compilation: ~7 min
- Total build time: ~22 min

### After Optimization (Both Builds)
- CUDA architectures: 1 (8.6 only)
- Gaussian compilation: ~3 min
- Total build time: ~15-18 min
- **Speedup: ~30-40%**

## Recommendations

### Use Single-Stage If
- Developing/debugging EGSTalker
- Need to compile additional CUDA extensions
- Disk space is not a concern
- Simpler Dockerfile is preferred

### Use Multi-Stage If
- Deploying to production
- Pushing to container registry
- Disk space is limited
- Security is a concern (fewer attack surfaces)
- Running multiple containers

## Testing Multi-Stage Build

```bash
# Build multi-stage
bash build_multistage.sh

# Test that it works
docker run --rm --gpus all egstalker:multistage python -c "
import torch
from diff_gaussian_rasterization import _C
from simple_knn import _C as simple_knn_C
print('All extensions loaded successfully!')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Check image size
docker images | grep egstalker
```

## Additional Optimizations Applied

1. **Removed redundant PyTorch installs**
   - Base image already has compatible torchvision/torchaudio

2. **Combined pip install layers**
   - Reduced from 7 RUN commands to 4
   - Better Docker layer caching

3. **Build arguments for flexibility**
   - `TORCH_CUDA_ARCH_LIST`: Customize GPU architectures
   - `MAX_JOBS`: Control parallel compilation (reduce if OOM)

## Reverting to Support Multiple GPUs

If you need to support different GPU types, change the ARG:

```dockerfile
# For A10 + A100:
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6"

# For T4 + A10:
ARG TORCH_CUDA_ARCH_LIST="7.5;8.6"

# For all modern GPUs (slower build):
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
```

Or pass at build time:
```bash
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6" -t egstalker .
```
