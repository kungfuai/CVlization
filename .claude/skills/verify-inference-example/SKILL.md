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

### Verification outcome and mode inventory

Before running commands, inventory every **primary inference mode** promised by
the task, `README.md`, `example.yaml`, shell presets, or wrapper CLI. A primary
mode is one users are told this example supports, including important model
variants when that variant is central to why the example exists. Do not infer
extra modes solely from the upstream model card.

Track evidence in a table and include it in the PR body or task report:

| Mode | Canonical input | Exact command | Result | Artifact | Inspected |
|------|-----------------|---------------|--------|----------|-----------|
| `<mode>` | `zzsi/cvl/<example>/...` or text | `...` | PASS/FAIL/SKIPPED | path/URL | what was checked |

Use these outcomes consistently:

- **VERIFIED**: every advertised primary mode and every applicable critical
  check in this skill passed. Outputs were inspected, not merely created.
- **PARTIAL**: any advertised primary mode or critical check failed or was
  skipped, even with a concrete blocker. Partial work may still be useful and
  may still open a draft PR, but it is not complete or verified.
- **UNVERIFIED**: inference was not run successfully.

Do not leave an untested mode advertised as verified. Either test it, narrow
the example's documented/metadata surface, or mark the result PARTIAL.

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

### 1b. Sample Data Management (CRITICAL)

Sample input files (images, videos, annotations, small weights) **must NOT be
committed to the git repo**. They are hosted on HuggingFace and lazy-downloaded
at runtime.

**Where sample data lives:**
- HuggingFace dataset repo: `zzsi/cvl` (centralized for all CVL examples)
- Files are organized under a per-example prefix: `<example_name>/...`
- Example: `ctrl_world/sample_data/droid_subset/videos/val/899/0.mp4`

**Required pattern in predict.py:**
```python
from huggingface_hub import hf_hub_download

HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "<example_name>"

def ensure_sample_data(cache_root=None):
    """Download sample data from HuggingFace if not cached."""
    if cache_root is None:
        cache_root = Path(os.environ.get(
            "HF_HOME", Path.home() / ".cache" / "huggingface"
        )) / "cvl_data" / "<example_name>"

    # Check if already downloaded
    if (cache_root / "some_marker_file").exists():
        return str(cache_root)

    for rel_path in FILES_TO_DOWNLOAD:
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_DATA_PREFIX}/{rel_path}",
            repo_type="dataset",
        )
        # Copy from HF cache to organized layout
        local_target = cache_root / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, local_target)

    return str(cache_root)
```

**What to verify:**
- `.gitignore` excludes `sample_data/`, `dataset_meta_info/`, and any local
  weight files (e.g. `*.pth`, `*.pt` patterns)
- `predict.py` has an `ensure_sample_data()` (or similar) function that
  downloads from `zzsi/cvl` with `repo_type="dataset"`
- Files are cached in `~/.cache/huggingface/cvl_data/<example_name>/` (inside
  the container this maps to the mounted HF cache)
- Second run skips download (check for "already cached" or no download messages)
- No sample data files are staged in git (`git status` shows no data files)
- Each binary sample has known provenance and redistribution permission. Do
  not upload arbitrary web images, voices, videos, or personal data.

**Uploading new sample data:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="local/path/to/file.jpg",
    path_in_repo="<example_name>/path/in/repo/file.jpg",
    repo_id="zzsi/cvl",
    repo_type="dataset",
)
```

### 1c. Upstream Source Strategy

When an example depends on an upstream research repository that is not
available as a suitable versioned package, review how its source enters the
example. A Dockerfile `git clone` is not automatically the best choice.

Choose deliberately:

- Use a released package when it exposes the required, compatible API.
- Consider vendoring a minimal, coherent inference subset when the code is
  small, permissively licensed, likely to need local compatibility fixes, and
  practical to test as part of CVlization.
- Retain an external checkout when the repository is large, changes rapidly,
  relies on submodules or native build machinery, or cannot be separated
  without creating a fragile partial copy.

For vendored source, prefer an example-local `upstream/` directory. The
example name is already supplied by its parent directory, so avoid redundant
layouts such as `vendor/<example_name>/`. Verify that:

- `upstream/UPSTREAM.md` records the repository URL, exact source commit,
  intentionally omitted paths, and CVlization modifications.
- Upstream copyright headers and all required `LICENSE` and `NOTICE` files are
  retained.
- The vendored tree excludes model weights, datasets, demo media, Git
  metadata, build outputs, and unrelated training or UI code.
- `predict.sh` exposes the committed `upstream/` tree through the example's
  runtime bind mount. The Dockerfile does not `COPY` either the CVlization
  wrapper or vendored source into the image.
- CVlization integration remains in the root wrapper when possible; upstream
  files are changed only when necessary and deviations are documented.
- The selected tree is a coherent runtime dependency, not copied functions
  whose hidden imports or configuration files were missed.

For an external checkout, pin an immutable commit, fetch only that revision,
and remove `.git` from the final image. Do not clone an unpinned branch or
perform a full-history clone followed by `git checkout`.

Changing between checkout and vendored source invalidates prior runtime
evidence. Rebuild and rerun every advertised primary mode before retaining
`VERIFIED`.

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

Repeat this section for every row in the primary-mode inventory. One successful
default smoke test does not verify other advertised modes such as image
conditioning, editing, streaming, voice cloning, alternate model variants, or
multimodal understanding.

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
- Output meaning matches the input and requested behavior. File existence,
  dimensions, duration, or byte size alone is not sufficient.

For visual, audio, and video outputs, inspect the media itself:

- Images: open them and check that they are nonblank, correctly framed, and
  visibly follow the prompt/input/edit.
- Video: play or render representative frames; check encoding, motion,
  temporal coherence, framing, and prompt/input adherence.
- Audio: listen to representative output; check intelligibility, duration,
  clipping/silence, and the requested speaker/style/content behavior.
- Text/JSON: read the result and assess semantic correctness against the input.

### 4a. Host Filesystem Persistence (CRITICAL)

**IMPORTANT**: When running via `cvl run`, outputs MUST be saved to the HOST filesystem (user's current working directory), not just inside the container. This is critical for usability.

**How it works:**
- User's cwd is mounted at `/mnt/cvl/workspace` inside the container
- `CVL_INPUTS` and `CVL_OUTPUTS` environment variables point to `/mnt/cvl/workspace`
- `predict.py` must use `resolve_output_path()` to resolve output paths
- `predict.py` must use `resolve_input_path()` to resolve input paths

**Verify host persistence:**
```bash
# Run from a test directory (NOT the example directory)
cd /tmp
mkdir -p cvl_test && cd cvl_test

# Run inference
cvl run <example-name> predict

# Check that outputs appear in current directory (NOT in example's outputs/ folder)
ls -la
# Expected: Output files appear here (e.g., output.mp4, result.json)

# Verify outputs are NOT ONLY in the container
ls -la /path/to/example/outputs/
# The container's outputs/ may also have files, but host cwd MUST have them
```

**What to verify:**
- Output files appear in user's current working directory when using `cvl run`
- `predict.py` imports and uses `resolve_output_path` from `cvlization.paths`
- `predict.py` imports and uses `resolve_input_path` for input file arguments
- Default output paths are resolved correctly (not hardcoded absolute container paths)

**Common issues:**
- **Outputs only in container**: Missing `resolve_output_path()` - outputs go to `/workspace/` or container's `outputs/`
- **Input files not found**: Missing `resolve_input_path()` - can't find files from user's cwd
- **Hardcoded paths**: Using `/workspace/output.mp4` instead of `resolve_output_path("output.mp4")`

**Required code patterns in predict.py:**
```python
from cvlization.paths import resolve_input_path, resolve_output_path

# For input files
input_path = resolve_input_path(args.input)  # Resolves against CVL_INPUTS

# For output files
output_path = resolve_output_path(args.output)  # Resolves against CVL_OUTPUTS

# For output directories
args.output_dir = resolve_output_path(args.output_dir.rstrip('/') + '/').rstrip('/')
```

### 4b. README Clarity — "What to Expect" (CRITICAL)

The README must give a first-time user a clear picture of what will happen when
they run `cvl run <name> predict`. Read the README and verify it answers all of
the following questions. If any are missing or unclear, update the README before
proceeding.

**Required information:**

1. **First-run cost**: Does the README mention model downloads and their total
   size? (e.g. "~17 GB download on first run, cached afterward")
2. **What it does**: Does the README concisely describe the inference task?
   (e.g. "replays 1 trajectory through 12 interaction steps")
3. **Where output goes**: Does the README state the output directory relative to
   the user's working directory? (e.g. "saves to `ctrl_world_outputs/` in your
   current directory")
4. **Output format**: Does the README describe what the output file looks like?
   (e.g. "MP4 video with 6-panel grid: 3 views × ground-truth / prediction")
5. **Runtime estimate**: Does the README give rough wall-clock time on at least
   one common GPU? (e.g. "~6 min on A100")

**How to verify:**
```bash
# Read the README and check for each item above
cat README.md

# Cross-check against predict.py defaults — the README should match
grep -E "default=" predict.py | head -20
```

**What to fix if missing:**
- Add a "What to expect" (or equivalent) section near the top of the README.
- Keep it factual — state defaults, sizes, and times; avoid marketing language.
- Make sure the section stays consistent with argparse defaults in predict.py
  (e.g. if default trajectory count or step count changes, README must match).

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
- Model weights remain outside the Docker image and download to a persistent
  mounted host cache: `~/.cache/cvlization` and/or the standard shared
  framework cache such as `~/.cache/huggingface/`
- Second run reuses cached models without re-downloading
- Second run also reuses canonical sample data without re-downloading
- Check predict.py doesn't set custom cache directories that break caching
- Record measured model/cache size and keep README estimates consistent with it

### 6. Runtime Checks

**GPU VRAM Usage Monitoring (REQUIRED for GPU models):**

Measure VRAM for every primary mode from model loading through inference and
shutdown. A one-time `nvidia-smi` query or manually watching `watch nvidia-smi`
does not establish a peak: short-lived allocations can occur between samples.

Use an otherwise-idle GPU and a polling interval of 100-250 ms. The wrapper
below records both device memory and the sum of compute-process memory on the
selected GPU UUID, stops its monitor on exit, and preserves the inference
command's exit status.

```bash
repo_root=$(git rev-parse --show-toplevel)
vram_monitor="$repo_root/.claude/skills/verify-inference-example/scripts/monitor_vram.sh"

# Confirm the selected GPU is idle and note its UUID and total memory.
nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader

# Repeat with the canonical command for every primary mode.
"$vram_monitor" /tmp/vram-<mode>.csv 0 -- \
    bash predict.sh <canonical-mode-arguments>
```

**Measurement validity rules:**
- Confirm the selected GPU has no competing compute processes before each run.
  If it is shared or becomes contaminated, do not subtract the baseline and
  call the result clean; wait or rerun on an idle GPU.
- Record an observed peak for every primary mode and materially different
  configuration. Include resolution, batch size, sequence/frame count,
  precision, quantization, offload settings, and shortened sampling steps.
- Keep the raw CSV until the PR is reviewed. Verify its sample count and check
  that no orphan monitor remains after an interrupted or failed attempt.
- Process memory is the sum of compute processes on the selected GPU. Device
  memory also includes driver/context overhead. Record both and identify which
  value is used in documentation.
- `nvidia-smi` reports memory values in MiB when `nounits` is used. Report the
  exact MiB value and optionally GiB (`MiB / 1024`). Do not divide MiB by 1000
  and label the result GB.
- Treat the result as the **observed peak for the tested configuration**, not a
  guaranteed minimum requirement. A run on a larger GPU does not prove that a
  smaller GPU works, even when the observed peak is below its nominal capacity.
- If monitoring fails or a primary mode is not measured, use `PARTIAL` rather
  than `VERIFIED`.

**Expected behavior:**
- **Model loading**: VRAM usage increases as model loads into memory
- **Inference peak**: VRAM spikes during forward pass
- **Cleanup**: Memory released after inference completes (for short-running containers)
- **Temperature**: Stable (<85°C)

**What to record for verification metadata:**
- GPU index, exact model, UUID, and total VRAM reported by `nvidia-smi`
- Idle baseline and post-run device memory
- Polling interval, sample count, and whether competing processes were absent
- Observed process and device peaks in MiB, plus correctly converted GiB
- Per-mode configuration that produced each peak
- Whether 4-bit/8-bit quantization was used (affects VRAM requirements)

**Troubleshooting:**
- **CUDA OOM**: Use smaller model variant, enable quantization (4-bit/8-bit), or run on CPU
- **High VRAM idle usage**: Check if other processes are using GPU
- **Memory not released**: Container may still be running (`docker ps`)
- **Implausible or duplicated samples**: Check for an orphan polling process,
  remove the invalid CSV, and rerun from an idle baseline

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

### 8. README Documentation Quality

The README should showcase concrete evidence that the example works, so a reader can evaluate it without running it.

**Required: Sample section** — add a `## Sample` section near the top with:

1. **Example input** — show what the model receives:
   - For image/video inputs: embed inline using a HuggingFace URL as the image source:
     ```markdown
     ![Sample input](https://huggingface.co/datasets/zzsi/cvl/resolve/main/<example-name>/sample_input.jpg)
     ```
   - Upload sample files to the `zzsi/cvl` HuggingFace dataset under `<example-name>/` if not already there
   - For text inputs: show a short excerpt in a code block

2. **Example output** — show a representative result:
   - For text/ekern/JSON/XML outputs: paste a meaningful excerpt in a fenced code block
   - For image/video outputs: embed with a HuggingFace URL (upload a sample output image to `zzsi/cvl`)
   - Include any key metrics (e.g., CER, accuracy) if available

Only upload **curated demo artifacts** that passed the semantic/perceptual
inspection above and are useful to a reader. Do not upload every run, failed or
low-quality outputs, large debug dumps, or artifacts without redistribution
permission. Use a stable per-example prefix and descriptive names, for example:

```text
<example-name>/sample_input.jpg
<example-name>/text_to_image_output.png
<example-name>/image_edit_output.png
<example-name>/vqa_result.json
```

**HuggingFace URL pattern** for raw file access (renders as `<img>` in GitHub Markdown):
```
https://huggingface.co/datasets/zzsi/cvl/resolve/main/<path-in-repo>
```

**Upload a file to zzsi/cvl:**
```python
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="/local/path/to/file.jpg",
    path_in_repo="<example-name>/sample_input.jpg",
    repo_id="zzsi/cvl",
    repo_type="dataset",
)
```

**Example Sample section:**
```markdown
## Sample

**Input** — piano score image (auto-downloaded):

![Sample score](https://huggingface.co/datasets/zzsi/cvl/resolve/main/smt_omr/sample_score.jpg)

**Output** — ekern notation (CER 0.28%):

​```
**ekern_1.0	**ekern_1.0
*clefF4	*clefG2
8FL	16r
...
​```
```

### 9. Update Verification Metadata

After successful verification, update the example.yaml with verification metadata:

**First, check GPU info:**
```bash
# Get GPU model and VRAM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

**Format:**
```yaml
verification:
  status: VERIFIED
  last_verified: 2025-10-25
  last_verification_note: >
    All primary modes verified on [GPU_MODEL] ([TOTAL_MIB] MiB total).
    Observed process peaks at 200 ms polling: mode_a [PEAK_MIB] MiB
    ([PEAK_GIB] GiB), mode_b [PEAK_MIB] MiB ([PEAK_GIB] GiB).
    Idle baseline [BASELINE_MIB] MiB; no competing processes.
  peak_vram_mib:
    mode_a: 12345
    mode_b: 23456
  vram_poll_interval_ms: 200
```

Use `status: PARTIAL` when any advertised primary mode or critical check is
failed/skipped, and name those gaps in `last_verification_note`. Use
`status: UNVERIFIED` when no inference mode completed successfully. Never use
`VERIFIED` merely because one smoke command succeeded.

**What to include in the note:**
- What was verified: build, inference, outputs
- Key aspects: model caching, GPU/CPU inference
- **GPU info**: Dynamically determine GPU model and total VRAM using
  `nvidia-smi`; preserve the reported MiB value
  - If no GPU: Use "CPU-only"
- **VRAM usage**: Per-mode observed process and device peaks from the polling
  CSV, including the baseline, interval, and tested configuration
  - Store exact values under `peak_vram_mib` when adding structured metadata
  - Convert MiB to GiB by dividing by 1024
  - Note if quantization (4-bit/8-bit) was used
- Any limitations: e.g., "Observed on a 48 GiB GPU; 24 GiB compatibility was
  not tested", "CUDA OOM with the documented full-resolution preset"
- Quick notes: e.g., "First run downloads 470MB models"

**Example complete entry:**
```yaml
name: pose-estimation-dwpose
docker: dwpose
capability: perception/pose_estimation
# ... other fields ...

verification:
  last_verified: 2025-10-25
  last_verification_note: "Verified build, inference with video/image inputs, model caching (470MB models), and JSON outputs on [detected GPU]."
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

# Issue: flash-attn build from source
# Fix: Never compile flash-attn in verification; use a prebuilt wheel that matches the image's
#      torch and CUDA versions (e.g., install via a direct wheel URL).

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

1. ✅ **Mode coverage**: Every advertised primary mode has exact PASS evidence
2. ✅ **Structure**: All required files present, example.yaml valid
3. ✅ **Sample Data**: Redistributable canonical inputs are hosted under the example prefix in `zzsi/cvl`, lazy-downloaded, and not committed to git
4. ✅ **Build**: Docker image builds without errors (both `./build.sh` and `cvl run <name> build`)
5. ✅ **Inference**: Every primary mode runs successfully on its canonical input
6. ✅ **Outputs**: Outputs are valid and semantically/perceptually inspected
7. ✅ **Host Persistence**: Outputs save to the host cwd through `cvl run`
8. ✅ **Input Resolution**: Custom inputs resolve correctly through `resolve_input_path()`
9. ✅ **Caching**: Models/data remain outside the image and a second run proves shared-cache reuse
10. ✅ **CVL CLI**: `cvl info <name>` and build/predict presets work
11. ✅ **Documentation**: README states first-run cost, behavior, output location/format, runtime, canonical inputs, and curated representative outputs
12. ✅ **Upstream Source**: Package, vendored source, or pinned checkout is deliberate and reproducible; vendored source is runtime-mounted, not copied into the image
13. ✅ **VRAM Evidence**: Every GPU mode has a clean, continuously sampled observed peak with exact units and tested configuration
14. ✅ **Verification Metadata**: `verification.status` matches the rules above and the note names exact coverage and limitations

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
- Use the polling wrapper above for GPU memory; a one-shot `nvidia-smi` sample
  is not peak evidence
- Check `docker logs <container>` if inference hangs
- For HuggingFace models, set `HF_TOKEN` environment variable if needed
- Most examples support custom input paths as arguments to predict.sh
- Check example.yaml for supported parameters and environment variables
- **For diffusion/flow matching models**: Reduce sampling steps for faster validation (e.g., `--num_steps 5` or `-i num_steps=5` for Cog). Most models support step parameters:
  - Common parameter names: `num_steps`, `num_inference_steps`, `steps`
  - Typical defaults: 20-50 steps
  - Fast validation: 5-10 steps (lower quality but completes quickly)
  - Production: Full step count for best quality
  - Examples: Stable Diffusion, SVD, FLUX, AnimateDiff, Flow Matching models
