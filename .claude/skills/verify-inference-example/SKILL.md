---
name: verify-inference-example
description: Verify a CVlization inference example is properly structured, builds successfully, and runs inference correctly. Use when validating inference example implementations or debugging inference issues.
---

# Verify Inference Example

Systematically verify that a CVlization inference example is complete, properly structured, and functional.

## When to Use

- Validating a new or modified inference example
- Debugging inference pipeline issues
- Ensuring example completeness before commits
- Verifying example works after CVlization updates

## Important Context

**Shared GPU Environment**: This machine may be used by multiple users simultaneously. Before running GPU-intensive inference:
1. Check GPU memory availability with `nvidia-smi`
2. Wait for sufficient VRAM and low GPU utilization if needed
3. Consider stopping other processes if you have permission
4. If CUDA OOM errors occur, wait and retry when GPU is less busy

## Verification Checklist

### 1. Structure Verification

Check that the example directory contains all required files:

```bash
# Navigate to example directory
cd examples/<capability>/<task>/<framework>/

# Expected structure:
# .
# ├── example.yaml        # Required: CVL metadata
# ├── Dockerfile          # Required: Container definition
# ├── build.sh            # Required: Build script
# ├── predict.sh          # Required: Inference script
# ├── predict.py          # Required: Inference code
# ├── examples/           # Required: Sample inputs
# ├── outputs/            # Created at runtime
# └── README.md           # Recommended: Documentation
```

**Key files to check:**
- `example.yaml` - Must have: name, capability, stability, presets (build, predict/inference)
- `Dockerfile` - Should copy necessary files and install dependencies
- `build.sh` - Must set `SCRIPT_DIR` and call `docker build`
- `predict.sh` - Must mount volumes correctly and call predict.py
- `predict.py` - Main inference script
- `examples/` - Directory with sample input files

### 2. Build Verification

```bash
# Option 1: Build using script directly
./build.sh

# Option 2: Build using CVL CLI (recommended)
cvl run <example-name> build

# Verify image was created
docker images | grep <example-name>

# Expected: Image appears with recent timestamp
```

**What to check:**
- Build completes without errors (both methods)
- All dependencies install successfully
- Image size is reasonable
- `cvl info <example-name>` shows correct metadata

### 3. Inference Verification

Run inference with sample inputs:

```bash
# Option 1: Run inference using script directly
./predict.sh

# Option 2: Run inference using CVL CLI (recommended)
cvl run <example-name> predict

# With custom inputs (if supported)
./predict.sh path/to/custom/input.jpg
```

**Immediate checks:**
- Container starts without errors
- Model loads successfully (check GPU memory with `nvidia-smi` if using GPU)
- Inference completes (outputs generated)
- Output files created in `outputs/` or similar directory
- Results look reasonable (open output files to inspect)

### 4. Output Verification

Check that inference produces valid outputs:

```bash
# Check outputs directory
ls -la outputs/

# Expected: Output files with recent timestamps

# Inspect output content
cat outputs/output.md  # For text outputs
# or
python -m json.tool outputs/output.json  # For JSON outputs
```

**What to verify:**
- Output files are created
- Output format is correct (markdown, JSON, etc.)
- Output contains expected content structure
- Output is non-empty and valid

### 5. Model Caching Verification

Verify that pretrained models are cached properly:

```bash
# Check HuggingFace cache
ls -la ~/.cache/huggingface/hub/
# Expected: Model files downloaded once and reused

# Run inference twice and verify no re-download
./predict.sh 2>&1 | tee first_run.log

# Second run should reuse cached models
./predict.sh 2>&1 | tee second_run.log

# Verify no download messages in second run
grep -i "downloading" second_run.log
# Expected: No new downloads (models already cached)
```

**What to verify:**
- Models download to `~/.cache/huggingface/` (or framework-specific cache)
- Second run reuses cached models without re-downloading
- Check predict.py doesn't set custom cache directories that break caching

### 6. Runtime Checks

**GPU Utilization (if applicable):**
```bash
# In another terminal
watch -n 1 nvidia-smi

# Expected:
# - GPU memory usage increases during model load
# - GPU utilization spikes during inference
# - Memory released after inference completes
```

**Docker Container Health:**
```bash
# Check container runs and exits cleanly
docker ps -a | head

# Verify mounts (for running container)
docker inspect <container-id> | grep -A 10 Mounts
# Should see: workspace, cvlization_repo, huggingface cache
```

### 7. Quick Validation Test

For fast verification during development:

```bash
# Run with smallest sample input
./predict.sh examples/small_sample.jpg

# Expected runtime: seconds to few minutes
# Verify: Completes without errors, output generated
```

### 8. Update Verification Metadata

After successful verification, update the example.yaml with verification metadata:

```bash
# Edit example.yaml to add verification field
# Add this at the end of the file (after all other fields)
```

**Format:**
```yaml
verification:
  last_verified: 2025-10-25
  last_verification_note: "Verified build, inference, model caching, and outputs"
```

**What to include in the note:**
- What was verified: build, inference, outputs
- Key aspects: model caching, GPU/CPU inference
- Any limitations: e.g., "Verified on A10 GPU only"
- Quick notes: e.g., "First run downloads 470MB models"

**Example complete entry:**
```yaml
name: pose-estimation-dwpose
docker: dwpose
capability: perception/pose_estimation
# ... other fields ...

verification:
  last_verified: 2025-10-25
  last_verification_note: "Verified build, inference with video/image inputs, model caching (470MB models), and JSON outputs. Runs on GPU."
```

**When to update:**
- After completing full verification checklist (steps 1-7)
- Only if ALL success criteria pass
- When re-verifying after CVlization updates or fixes

## Common Issues and Fixes

### Build Failures
```bash
# Issue: Dockerfile can't find files
# Fix: Check COPY paths are relative to Dockerfile location

# Issue: Dependency conflicts
# Fix: Check requirements.txt versions, update base image

# Issue: Large build context
# Fix: Add .dockerignore file
```

### Inference Failures
```bash
# Issue: CUDA out of memory
# Fix: Use smaller model variant or CPU inference

# Issue: Model not found
# Fix: Check model name/path in predict.py, ensure internet connection

# Issue: Input file not found
# Fix: Check file paths, ensure examples/ directory exists

# Issue: Permission denied on outputs
# Fix: Ensure output directories exist and are writable
```

### Output Issues
```bash
# Issue: Empty outputs
# Fix: Check model loaded correctly, verify input format

# Issue: Malformed JSON output
# Fix: Check output parsing logic in predict.py

# Issue: Outputs not saved
# Fix: Verify output directory path, check file write permissions
```

## Example Commands

### Document AI - Granite Docling
```bash
cd examples/perception/doc_ai/granite_docling
./build.sh
./predict.sh
# Check: outputs/output.md contains extracted document structure
```

### Vision-Language - Moondream
```bash
cd examples/perception/vision_language/moondream2
./build.sh
./predict.sh examples/demo.jpg
# Check: outputs/ contains image description
```

## CVL Integration

Inference examples integrate with CVL command system:

```bash
# List all available examples
cvl list

# Get example info
cvl info granite-docling

# Run example directly (uses example.yaml presets)
cvl run granite-docling build
cvl run granite-docling predict
```

## Success Criteria

An inference example passes verification when:

1. ✅ **Structure**: All required files present, example.yaml valid
2. ✅ **Build**: Docker image builds without errors (both `./build.sh` and `cvl run <name> build`)
3. ✅ **Inference**: Runs successfully on sample inputs (both `./predict.sh` and `cvl run <name> predict`)
4. ✅ **Outputs**: Valid output files generated in expected format
5. ✅ **Model Caching**: Models cached to `~/.cache/` (typically `~/.cache/huggingface/`), avoiding repeated downloads
6. ✅ **CVL CLI**: `cvl info <name>` shows correct metadata, build and predict presets work
7. ✅ **Documentation**: README explains how to use the example
8. ✅ **Verification Metadata**: example.yaml updated with `verification` field containing `last_verified` date and `last_verification_note`

## Related Files

Check these files for debugging:
- `predict.py` - Core inference logic
- `predict.sh` - Docker run script
- `Dockerfile` - Environment setup
- `example.yaml` - CVL metadata and presets
- `examples/` - Sample input files
- `README.md` - Usage instructions

## Tips

- Use small sample inputs for fast validation
- Monitor GPU memory with `nvidia-smi` if using GPU
- Check `docker logs <container>` if inference hangs
- For HuggingFace models, set `HF_TOKEN` environment variable if needed
- Most examples support custom input paths as arguments to predict.sh
- Check example.yaml for supported parameters and environment variables
