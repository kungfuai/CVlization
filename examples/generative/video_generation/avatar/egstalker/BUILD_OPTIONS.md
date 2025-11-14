# Build Options for EGSTalker Docker Image

## Quick Reference

```bash
# Default build (4 CPU cores, quiet)
bash build.sh

# Verbose build with compilation logs
VERBOSE=1 bash build.sh

# Fast build with 8 CPU cores
MAX_JOBS=8 bash build.sh

# Verbose + fast (recommended for debugging)
VERBOSE=1 MAX_JOBS=8 bash build.sh
```

## 1. Verbose Compilation Logs

**Problem**: By default, Docker buildkit hides compilation output
**Solution**: Use `VERBOSE=1` to see all gcc/nvcc commands

```bash
# See full compilation logs
VERBOSE=1 bash build.sh
```

This adds `--progress=plain` to the docker build command, showing:
- All pip install progress
- gcc compilation commands for CUDA code
- nvcc compilation for GPU kernels
- Build warnings and errors in real-time

**When to use**:
- Debugging build failures
- Understanding what's being compiled
- Monitoring long compilations (pytorch3d, Gaussian rasterization)

## 2. Parallel Compilation (MAX_JOBS)

**Problem**: pytorch3d and Gaussian rasterization compile slowly with default settings
**Solution**: Use more CPU cores with `MAX_JOBS`

```bash
# Use 8 CPU cores (faster)
MAX_JOBS=8 bash build.sh

# Use all available cores
MAX_JOBS=$(nproc) bash build.sh

# Conservative setting (less RAM usage)
MAX_JOBS=4 bash build.sh
```

**Speed Comparison** (pytorch3d compilation):
- `MAX_JOBS=1`: ~15 minutes
- `MAX_JOBS=4`: ~5-7 minutes (default)
- `MAX_JOBS=8`: ~3-4 minutes
- `MAX_JOBS=16`: ~2-3 minutes (diminishing returns)

**System Requirements**:
- Each job uses ~2-3GB RAM during compilation
- `MAX_JOBS=8` needs ~16-24GB RAM available
- Monitor with: `htop` or `top` during build

## 3. Build Arguments (Advanced)

For complete control, use docker build directly:

```bash
# Custom CUDA architecture
docker build \
    --progress=plain \
    --build-arg TORCH_CUDA_ARCH_LIST="8.6;9.0" \
    --build-arg MAX_JOBS=8 \
    -t egstalker \
    .

# Override build args via build script
VERBOSE=1 MAX_JOBS=16 bash build.sh
```

## 4. Multi-Stage Build

Same options work for the multi-stage build:

```bash
# Verbose multi-stage build
VERBOSE=1 bash build_multistage.sh

# Fast multi-stage build
MAX_JOBS=8 bash build_multistage.sh

# Both
VERBOSE=1 MAX_JOBS=8 bash build_multistage.sh
```

## How It Works

### Before (pytorch3d used 1 core):
```dockerfile
RUN pip install pytorch3d
ENV MAX_JOBS=4  # ❌ Too late!
```

### After (pytorch3d uses MAX_JOBS cores):
```dockerfile
ENV MAX_JOBS=4  # ✅ Set before compilation
RUN pip install pytorch3d
```

The `MAX_JOBS` environment variable is respected by:
- `pytorch3d` setup.py
- `diff-gaussian-rasterization` CUDA compilation
- `simple-knn` CUDA compilation

## Troubleshooting

### Build runs out of memory
```bash
# Reduce parallel jobs
MAX_JOBS=2 bash build.sh
```

### Can't see what's compiling
```bash
# Enable verbose output
VERBOSE=1 bash build.sh
```

### Build is taking too long
```bash
# Check CPU usage
htop

# If CPUs not maxed out, increase MAX_JOBS
MAX_JOBS=8 bash build.sh
```

### Check CPU count
```bash
nproc              # Total cores
nproc --all        # Including hyperthreading
free -h            # Check available RAM
```

## Recommended Settings

**For this A10 machine**:
```bash
# Check system resources
nproc    # 48 cores
free -h  # 185Gi RAM

# Recommended for fast builds
MAX_JOBS=16 VERBOSE=1 bash build.sh
```

**General guidelines**:
- **Development**: `VERBOSE=1 MAX_JOBS=4` (see what's happening)
- **Production**: `MAX_JOBS=8` (fast, reliable)
- **Maximum speed**: `MAX_JOBS=$(nproc)` (use all cores)
- **Low memory**: `MAX_JOBS=2` (4-8GB RAM available)
