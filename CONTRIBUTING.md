# Contributing to CVlization

Thank you for contributing to CVlization! This guide documents the standardization patterns used across all examples.

## Contribution Workflow

### Adding a New Dockerized Example

1. **Research the upstream project**
   ```bash
   # Clone the research repo you want to base the example on
   git clone https://github.com/original/research-repo /tmp/research-repo
   cd /tmp/research-repo
   ```
   - Understand the codebase well
   - Estimate the complexity of vendoring the code
   - Check license compatibility
   - If complexity is manageable, proceed

2. **Find a similar example** in CVlization to use as a template
   ```bash
   # Browse examples with similar patterns
   ls examples/generative/video_generation/avatar/  # for avatar examples
   ls examples/perception/doc_ai/     # for document AI examples
   ```

3. **Choose the right category** for your example:
   - `perception/` - Understanding signals (vision, speech, multimodal)
   - `generative/` - Creating content (text, images, video, audio)
   - `analytical/` - Prediction & forecasting (time series, tabular)
   - `physical/` - Robotics & embodied AI
   - `agentic/` - AI agents (RAG, tool use, workflows)

3. **Create your example directory**:
   ```bash
   mkdir -p examples/<category>/<subcategory>/<your_example>
   cd examples/<category>/<subcategory>/<your_example>
   ```

4. **Create the required files** (see [Example Structure](#example-structure) below):
   - `example.yaml` - Metadata and presets
   - `Dockerfile` - Container definition
   - `build.sh` - Build script
   - `predict.sh` (or `train.sh`) - Main functionality
   - `README.md` - Documentation

5. **Focus on `build` and `predict` first**
   - Get the Dockerfile working with pinned dependencies
   - Implement `predict.py` and necessary modules
   - Use **centralized caching** for models (`~/.cache/cvlization` → `/root/.cache`)
   - Use **lazy downloading** - download models on first run, not during build
   - Add training support later if applicable

6. **Test and verify carefully**:
   ```bash
   pip install -e .  # Install cvl CLI
   cvl run <your_example> build
   cvl run <your_example> predict
   ```
   You can use the Claude Code skills in this repo for automated verification:
   - `verify-inference-example` - For inference-only examples
   - `verify-training-pipeline` - For training examples

7. **Submit a Pull Request** with:
   - Clear description of what the example does
   - Any license considerations (see [License Notes](#license-notes))
   - Verification status (did it build and run successfully?)

### License Notes

Each example may package open-source projects with different licenses. When contributing:
- Check the license of any model/code you're packaging
- Add a `LICENSE` file in your example directory if needed
- Note any usage restrictions in your README.md

## Example Structure

Each dockerized example follows this standard structure:

```
examples/<task>/<example_name>/
├── example.yaml          # Required: Example metadata and presets
├── Dockerfile            # Required: Container definition
├── build.sh              # Required: Build the Docker image
├── <preset>.sh           # Optional: Additional preset scripts (train.sh, predict.sh, etc.)
└── README.md             # Recommended: Example documentation
```

## example.yaml Format

All examples must include an `example.yaml` file with standardized metadata:

```yaml
name: example_name
capability: category/subcategory  # e.g., generative/llm, perception/ocr_and_layout
modalities:
  - text
  - vision
datasets:
  - dataset_name
stability: stable|beta|experimental
resources:
  gpu: 1              # Number of GPUs required
  vram_gb: 8          # VRAM per GPU in GB
  disk_gb: 2          # Disk space required in GB
presets:
  build:
    script: build.sh
    description: Build the Docker image
  train:
    script: train.sh
    description: Train the model
  predict:
    script: predict.sh
    description: Run inference
tags:
  - tag1
  - tag2
tasks:
  - task_name
frameworks:
  - pytorch|tensorflow|jax
description: Brief description of what this example does
```

### Preset Format (Required)

The `presets` section **must** use the dict format with explicit script names and descriptions:

```yaml
presets:
  build:
    script: build.sh
    description: Build the Docker image
  train:
    script: train.sh
    description: Train the model
```

**Do not** use the legacy list format:
```yaml
# ❌ Old format (deprecated)
presets:
  - build
  - train
```

### Required Presets

All dockerized examples **must** include a `build` preset:

```yaml
presets:
  build:
    script: build.sh
    description: Build the Docker image
```

## Shell Script Pattern

All shell scripts (`.sh` files) must use `SCRIPT_DIR` pattern to avoid fragile relative paths:

```bash
#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define cache directory (centralized)
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

# Use the paths
docker run \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    your-image-name
```

### Do NOT use fragile relative paths:

```bash
# ❌ Fragile - breaks if script is called from different directory
REPO_ROOT="../../../"
CACHE_DIR="../../../data/container_cache"
```

## Docker Image Naming

Docker images are named after the example directory:

- Example path: `examples/generative/llm/nanogpt/`
- Image name: `nanogpt`

The `build.sh` script should build an image matching the directory name:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

docker build -t "$IMAGE_NAME" .
```

## Cache Management

Examples use centralized caching to avoid re-downloading models and datasets:

- **Host path**: `~/.cache/cvlization/`
- **Container path**: `/root/.cache` (standard cache location)

Mount the cache directory in your Docker run commands:

```bash
CACHE_DIR="${HOME}/.cache/cvlization"
mkdir -p "$CACHE_DIR"

docker run \
    -v "$CACHE_DIR:/root/.cache" \
    your-image-name
```

The centralized cache automatically works with HuggingFace, PyTorch, and other ML frameworks.

## User Permissions

Always run containers as the host user to avoid root-owned files:

```bash
docker run \
    --user "$(id -u):$(id -g)" \
    -v "$CACHE_DIR:/root/.cache" \
    your-image-name
```

## Example: Complete build.sh

```bash
#!/bin/bash
set -e

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Build image
docker build -t "$IMAGE_NAME" .

echo "Built image: $IMAGE_NAME"
```

## Example: Complete train.sh

```bash
#!/bin/bash
set -e

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${HOME}/.cache/cvlization"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# Ensure cache directory exists
mkdir -p "$CACHE_DIR"

# Run training
docker run \
    --gpus=all \
    --user "$(id -u):$(id -g)" \
    -v "$CACHE_DIR:/root/.cache" \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    python train.py "$@"
```

## Testing Your Example

Before submitting, verify your example follows the pattern:

1. **Check example.yaml format**:
   ```bash
   # Should have 'build' preset in dict format
   grep -A2 "presets:" examples/your_example/example.yaml
   ```

2. **Check script paths**:
   ```bash
   # Should use SCRIPT_DIR pattern, not ../../../
   grep "SCRIPT_DIR" examples/your_example/*.sh
   grep -c "\.\./\.\./\.\." examples/your_example/*.sh  # Should be 0
   ```

3. **Test with cvl CLI**:
   ```bash
   cvl info your_category/your_example
   cvl run your_category/your_example build
   cvl run your_category/your_example train
   ```

## Stability Levels

Mark your example's stability in `example.yaml`:

- **stable**: Production-ready, well-tested, actively maintained
- **beta**: Working but may have rough edges or limited testing
- **experimental**: Proof of concept, may not work reliably

## Checklist for New Examples

- [ ] `example.yaml` with all required fields
- [ ] `presets.build` using dict format (not list)
- [ ] `build.sh` script that builds Docker image
- [ ] Scripts use `SCRIPT_DIR` pattern for paths (no `../../../`)
- [ ] Docker image named after directory
- [ ] Cache mounted to `/root/.cache` from `~/.cache/cvlization`
- [ ] Containers run as host user (`--user $(id -u):$(id -g)`)
- [ ] `README.md` with usage instructions
- [ ] Tested with `cvl run` CLI

## Verification

CVlization includes Claude Code skills for automated verification:

- **`verify-training-pipeline`** - Validates training examples end-to-end
- **`verify-inference-example`** - Validates inference examples

Add verification metadata to your `example.yaml`:

```yaml
verification:
  last_verified: "2025-12-15"
  last_verification_note: "Verified build, inference, and model caching on A10 GPU (24GB VRAM). Training runs successfully with loss decreasing properly."
```

## Questions?

Open an issue on [GitHub](https://github.com/kungfuai/CVlization/issues) or check existing examples for patterns.
