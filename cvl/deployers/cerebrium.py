"""Cerebrium serverless deployment."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any


# Cerebrium GPU options (from docs)
# GPU Model         Identifier          VRAM    Plan required
# NVIDIA H100       HOPPER_H100         80GB    Enterprise
# NVIDIA A100 80GB  AMPERE_A100_80GB    80GB    Enterprise
# NVIDIA A100 40GB  AMPERE_A100_40GB    40GB    Enterprise
# NVIDIA L40s       ADA_L40             48GB    Hobby+
# NVIDIA L4         ADA_L4              24GB    Hobby+
# NVIDIA A10        AMPERE_A10          24GB    Hobby+
# NVIDIA T4         TURING_T4           16GB    Hobby+

CEREBRIUM_GPUS = {
    "T4": {"id": "TURING_T4", "vram": 16, "plan": "Hobby+"},
    "L4": {"id": "ADA_L4", "vram": 24, "plan": "Hobby+"},
    "A10": {"id": "AMPERE_A10", "vram": 24, "plan": "Hobby+"},
    "L40": {"id": "ADA_L40", "vram": 48, "plan": "Hobby+"},
    "A100": {"id": "AMPERE_A100_40GB", "vram": 40, "plan": "Enterprise"},
    "A100_80GB": {"id": "AMPERE_A100_80GB", "vram": 80, "plan": "Enterprise"},
    "H100": {"id": "HOPPER_H100", "vram": 80, "plan": "Enterprise"},
}

# Map VRAM requirements to recommended GPU (prefer Hobby+ tier when possible)
VRAM_TO_GPU = {
    16: "T4",        # 16GB VRAM
    24: "A10",       # 24GB VRAM (A10 or L4)
    40: "L40",       # 48GB VRAM (L40s is Hobby+)
    48: "L40",       # 48GB VRAM
    80: "A100_80GB", # 80GB VRAM (Enterprise)
}

# Examples with automatic main.py generation support
SUPPORTED_EXAMPLES = {"ltx2"}

# Models required for each example (HuggingFace repo IDs and optional file list)
# These are uploaded to Cerebrium persistent storage during deploy
# Format: repo_id -> list of files (None means full repo)
EXAMPLE_MODELS = {
    "ltx2": {
        "Lightricks/LTX-2": [  # Only upload files needed for distilled fp8 pipeline (~27GB)
            "ltx-2-19b-distilled-fp8.safetensors",
            "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        ],
        "google/gemma-3-12b-it-qat-q4_0-unquantized": None,  # Full repo (~23GB)
    },
}

# Cerebrium persistent storage path for HuggingFace cache
CEREBRIUM_HF_CACHE = ".cache/huggingface/hub"


def get_cerebrium_gpu(vram_gb: int, gpu_override: Optional[str] = None) -> tuple[str, str]:
    """
    Map VRAM requirement to Cerebrium GPU type.

    Args:
        vram_gb: Required VRAM in GB
        gpu_override: User-specified GPU type override

    Returns:
        (gpu_identifier, gpu_short_name) e.g. ("AMPERE_A10", "A10")
    """
    if gpu_override:
        if gpu_override in CEREBRIUM_GPUS:
            gpu_info = CEREBRIUM_GPUS[gpu_override]
            return gpu_info["id"], gpu_override
        else:
            # Assume it's already a full identifier
            return gpu_override, gpu_override

    # Auto-select based on VRAM
    for threshold, gpu_name in sorted(VRAM_TO_GPU.items()):
        if vram_gb <= threshold:
            return CEREBRIUM_GPUS[gpu_name]["id"], gpu_name

    # Default to H100 for very large models
    return CEREBRIUM_GPUS["H100"]["id"], "H100"


class CerebriumDeployer:
    """Deploy CVL examples to Cerebrium serverless platform."""

    def __init__(
        self,
        example_path: Path,
        example_meta: Dict[str, Any],
        gpu_override: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Initialize deployer.

        Args:
            example_path: Path to example directory
            example_meta: Example metadata from example.yaml
            gpu_override: Override GPU type (e.g., "A10", "A100", "H100")
            project_id: Cerebrium project ID
        """
        self.example_path = example_path
        self.example_meta = example_meta
        self.name = example_meta.get("name", example_path.name)
        self.gpu_override = gpu_override
        self.project_id = project_id

    def is_supported(self) -> bool:
        """Check if this example has automatic deployment support."""
        return self.name in SUPPORTED_EXAMPLES

    def get_unsupported_message(self) -> str:
        """Get message explaining why example is not supported."""
        return (
            f"Automatic Cerebrium deployment not yet supported for '{self.name}'.\n"
            f"Currently supported: {', '.join(sorted(SUPPORTED_EXAMPLES))}"
        )

    def _find_cli(self) -> Optional[str]:
        """Find cerebrium CLI binary path."""
        return shutil.which("cerebrium")

    def check_cli(self) -> tuple[bool, str]:
        """Check if Cerebrium CLI is installed and authenticated.

        Returns:
            (is_ready, message)
        """
        # Check if CLI is installed
        cli_path = self._find_cli()
        if not cli_path:
            return False, (
                "Cerebrium CLI not found.\n"
                "Install with: uv add cerebrium  (or: pip install cerebrium)\n"
                "Then run: cerebrium login"
            )

        print(f"  Found: {cli_path}", flush=True)

        # Check if logged in by checking config file
        config_path = Path.home() / ".cerebrium" / "config.yaml"
        if not config_path.exists():
            # Also check for CEREBRIUM_TOKEN env var (CI mode)
            if not os.environ.get("CEREBRIUM_TOKEN"):
                return False, (
                    "Not logged in to Cerebrium.\n"
                    "  Run: cerebrium login\n"
                    "  Or set CEREBRIUM_TOKEN environment variable for CI/CD"
                )

        return True, "Cerebrium CLI ready"

    def get_gpu_config(self) -> tuple[str, str, int]:
        """
        Get GPU configuration for deployment.

        Returns:
            (gpu_identifier, gpu_short_name, vram_gb)
        """
        resources = self.example_meta.get("resources", {})
        vram_gb = resources.get("vram_gb", 24)
        gpu_id, gpu_name = get_cerebrium_gpu(vram_gb, self.gpu_override)
        return gpu_id, gpu_name, vram_gb

    def get_required_models(self) -> dict[str, list[str] | None]:
        """Get dict of HuggingFace repo IDs -> file lists (None means full repo)."""
        return EXAMPLE_MODELS.get(self.name, {})

    def _get_hf_cache_path(self, repo_id: str) -> Path:
        """Get local HuggingFace cache path for a repo."""
        # HF cache format: models--{org}--{repo}
        cache_name = "models--" + repo_id.replace("/", "--")
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        return Path(hf_home) / "hub" / cache_name

    def _check_cerebrium_storage(self, repo_id: str, files: list[str] | None = None) -> bool:
        """Check if model exists and is complete in Cerebrium persistent storage.

        Checks for files in snapshots/ directory (where hf_hub_download looks).
        """
        cache_name = "models--" + repo_id.replace("/", "--")
        snapshots_path = f"{CEREBRIUM_HF_CACHE}/{cache_name}/snapshots"

        result = subprocess.run(
            ["cerebrium", "ls", snapshots_path],
            capture_output=True,
            text=True,
        )
        # cerebrium ls returns 0 even for non-existent paths, but prints "No files found"
        if result.returncode != 0:
            return False
        if "No files found" in result.stdout or "No files found" in result.stderr:
            return False

        # If specific files requested, check they exist in the snapshot
        if files:
            # List files in the snapshot revision directory
            # First, find the revision directory
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l and not l.startswith("NAME")]
            if not lines:
                return False

            # Get first revision directory name (remove trailing /)
            revision_dir = lines[0].split()[0].rstrip("/")
            revision_path = f"{snapshots_path}/{revision_dir}"

            result = subprocess.run(
                ["cerebrium", "ls", revision_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or "No files found" in result.stdout:
                return False

            # Check each requested file exists
            existing_files = result.stdout
            for filename in files:
                if filename not in existing_files:
                    return False

            # Check for incomplete uploads
            if ".incomplete" in existing_files:
                return False

        return True

    def _download_model(self, repo_id: str, files: list[str] | None = None) -> Path:
        """Download model (or specific files) to local HuggingFace cache."""
        local_path = self._get_hf_cache_path(repo_id)

        if files is None:
            # Full repo download
            if local_path.exists() and (local_path / "snapshots").exists():
                print(f"    Model already in local cache: {repo_id}")
                return local_path

            print(f"    Downloading {repo_id} (full repo)...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id)
                print(f"    Downloaded: {repo_id}")
            except Exception as e:
                print(f"    Warning: Failed to download {repo_id}: {e}")
                raise
        else:
            # Specific files download
            print(f"    Downloading {len(files)} files from {repo_id}...")
            try:
                from huggingface_hub import hf_hub_download
                for filename in files:
                    file_path = local_path / "snapshots"
                    # Check if file already exists in any snapshot
                    already_exists = False
                    if file_path.exists():
                        for snapshot in file_path.iterdir():
                            if (snapshot / filename).exists():
                                already_exists = True
                                break

                    if already_exists:
                        print(f"      {filename}: already cached")
                    else:
                        print(f"      {filename}: downloading...")
                        hf_hub_download(repo_id=repo_id, filename=filename)
                print(f"    Downloaded: {len(files)} files")
            except Exception as e:
                print(f"    Warning: Failed to download from {repo_id}: {e}")
                raise

        return local_path

    def _upload_model_to_cerebrium(self, repo_id: str, local_path: Path, files: list[str] | None = None) -> bool:
        """Upload model from local cache to Cerebrium persistent storage.

        Uploads refs + snapshots directories. Symlinks in snapshots are resolved
        to actual files by cerebrium cp, so hf_hub_download() finds them directly.
        Blobs are skipped since snapshots contains the resolved file content.
        """
        cache_name = "models--" + repo_id.replace("/", "--")
        remote_base = f"{CEREBRIUM_HF_CACHE}/{cache_name}"

        print(f"    Uploading {repo_id} to Cerebrium...")
        print(f"      Local: {local_path}")
        print(f"      Remote: /persistent-storage/{remote_base}")

        # Upload refs directory (contains revision hash)
        refs_dir = local_path / "refs"
        if refs_dir.exists():
            print(f"      Uploading refs...")
            result = subprocess.run(
                ["cerebrium", "cp", str(refs_dir), f"{remote_base}/refs"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"    Warning: refs upload failed: {result.stderr}")
                return False

        # Upload snapshots directory (symlinks resolved to actual files)
        # This is where hf_hub_download() looks for cached files
        snapshots_dir = local_path / "snapshots"
        if not snapshots_dir.exists():
            print(f"    Warning: No snapshots directory found")
            return False

        # If specific files requested, only upload those from the snapshot
        if files:
            # Find the latest snapshot revision
            snapshot_revisions = sorted(
                snapshots_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not snapshot_revisions:
                print(f"    Warning: No snapshot revisions found")
                return False

            latest_snapshot = snapshot_revisions[0]
            revision = latest_snapshot.name

            # Calculate total size for progress
            total_size = 0
            for filename in files:
                file_path = latest_snapshot / filename
                if file_path.exists() or file_path.is_symlink():
                    # Resolve symlink to get actual size
                    real_path = file_path.resolve()
                    if real_path.exists():
                        total_size += real_path.stat().st_size

            print(f"      Uploading {len(files)} files ({total_size / 1e9:.1f}GB) from snapshot {revision[:8]}...")

            # Upload each file individually
            for filename in files:
                file_path = latest_snapshot / filename
                if not file_path.exists() and not file_path.is_symlink():
                    print(f"    Warning: File not found in snapshot: {filename}")
                    return False

                # Resolve symlink to get actual file path
                real_path = file_path.resolve()
                if not real_path.exists():
                    print(f"    Warning: Resolved file not found: {real_path}")
                    return False

                remote_path = f"{remote_base}/snapshots/{revision}/{filename}"
                size_gb = real_path.stat().st_size / 1e9

                print(f"      Uploading: {filename} ({size_gb:.1f}GB)")
                result = subprocess.run(
                    ["cerebrium", "cp", str(real_path), remote_path],
                    text=True,
                )
                if result.returncode != 0:
                    print(f"    Warning: Upload failed for {filename}")
                    return False
        else:
            # Upload entire snapshots directory
            print(f"      Uploading snapshots/ (symlinks will be resolved)...")
            result = subprocess.run(
                ["cerebrium", "cp", str(snapshots_dir), f"{remote_base}/snapshots"],
                text=True,
            )
            if result.returncode != 0:
                print(f"    Warning: snapshots upload failed")
                return False

        print(f"    Uploaded: {repo_id}")
        return True

    def upload_models(self) -> dict[str, bool]:
        """
        Upload required models to Cerebrium persistent storage.

        Downloads models to local cache if not present, then uploads to Cerebrium.

        Returns:
            Dict of repo_id -> success status
        """
        models = self.get_required_models()
        if not models:
            print("No models to upload for this example")
            return {}

        # Check for huggingface_hub upfront (needed if any model requires download)
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("\nError: huggingface_hub is required for model upload.")
            print("Install with: pip install huggingface_hub")
            print("Or: pip install cvl[deploy]")
            print("\nAlternatively, use --skip-model-upload and upload models manually.")
            return {repo_id: False for repo_id in models}

        print(f"\nUploading models to Cerebrium persistent storage...")
        results = {}

        for repo_id, files in models.items():
            if files:
                print(f"  Processing: {repo_id} ({len(files)} files)")
            else:
                print(f"  Processing: {repo_id} (full repo)")

            # Check if already in Cerebrium
            if self._check_cerebrium_storage(repo_id, files):
                print(f"    Already in Cerebrium storage, skipping")
                results[repo_id] = True
                continue

            # Download to local cache if needed
            try:
                local_path = self._download_model(repo_id, files)
            except Exception:
                results[repo_id] = False
                continue

            # Upload to Cerebrium
            success = self._upload_model_to_cerebrium(repo_id, local_path, files)
            results[repo_id] = success

        # Summary
        uploaded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        print(f"\nModel upload complete: {uploaded} succeeded, {failed} failed")

        return results

    def _parse_dockerfile(self) -> tuple[Optional[str], list[str]]:
        """Parse Dockerfile to extract base image and apt packages.

        Returns:
            (base_image, apt_packages)
        """
        dockerfile = self.example_path / "Dockerfile"
        if not dockerfile.exists():
            return None, []

        base_image = None
        apt_packages = []

        content = dockerfile.read_text()
        lines = content.splitlines()

        # Join lines with backslash continuations
        joined_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Keep joining lines while current ends with backslash
            while line.rstrip().endswith("\\") and i + 1 < len(lines):
                line = line.rstrip()[:-1] + " " + lines[i + 1].strip()
                i += 1
            joined_lines.append(line)
            i += 1

        for line in joined_lines:
            line = line.strip()

            # Extract FROM base image
            if line.upper().startswith("FROM "):
                base_image = line.split()[1]

            # Extract apt-get install packages
            if "apt-get install" in line:
                # Parse packages from apt-get install command
                parts = line.split("apt-get install")[-1]
                # Remove flags like -y, --no-install-recommends
                for token in parts.split():
                    if token.startswith("-"):
                        continue
                    if token in ("&&", "||", ";", "rm", "apt-get", "update"):
                        break
                    if token and not token.startswith("/"):
                        apt_packages.append(token)

        return base_image, apt_packages

    def _parse_requirements(self) -> dict[str, str]:
        """Parse requirements.txt to extract pip packages.

        Returns:
            Dict of package_name -> version_spec
        """
        requirements = self.example_path / "requirements.txt"
        if not requirements.exists():
            return {}

        packages = {}
        for line in requirements.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Parse package==version or package>=version etc.
            for op in ["==", ">=", "<=", "~=", "!="]:
                if op in line:
                    name, version = line.split(op, 1)
                    packages[name.strip()] = f"{op}{version.strip()}"
                    break
            else:
                # No version specifier
                packages[line] = "latest"

        return packages

    def generate_cerebrium_toml(self, deploy_dir: Path) -> str:
        """Generate cerebrium.toml configuration for custom Dockerfile runtime."""
        resources = self.example_meta.get("resources", {})
        vram_gb = resources.get("vram_gb", 24)
        gpu_count = resources.get("gpu_count", resources.get("gpu", 1))

        gpu_id, gpu_name, _ = self.get_gpu_config()

        # Estimate CPU memory: 1.5x VRAM for large model loading overhead
        memory_gb = max(32, int(vram_gb * 1.5))

        # Build toml content for custom Dockerfile runtime
        toml_lines = [
            "[cerebrium.deployment]",
            f'name = "{self.name}"',
        ]

        # Add project_id if provided
        if self.project_id:
            toml_lines.append(f'project_id = "{self.project_id}"')

        toml_lines.extend([
            "",
            "[cerebrium.hardware]",
            f'compute = "{gpu_id}"',
            f"gpu_count = {gpu_count}",
            "cpu = 4",
            f"memory = {memory_gb}",
            "",
            "[cerebrium.scaling]",
            "min_replicas = 0",
            "max_replicas = 5",
            "cooldown = 300",
            "",
            "# Custom Dockerfile runtime - we use our own Docker image",
            "# This gives full control over dependencies and paths",
            "[cerebrium.runtime.custom]",
            "port = 8192",
            'healthcheck_endpoint = "/health"',
            'readycheck_endpoint = "/ready"',
            'dockerfile_path = "./Dockerfile"',
        ])

        toml_content = "\n".join(toml_lines) + "\n"
        toml_path = deploy_dir / "cerebrium.toml"
        toml_path.write_text(toml_content)
        return str(toml_path)

    def generate_main_py(self, deploy_dir: Path) -> str:
        """Generate main.py FastAPI entry point for Cerebrium custom runtime.

        Currently only supports ltx2 example. Other examples will need
        manual main.py creation or future implementation.
        """
        if self.name != "ltx2":
            raise NotImplementedError(
                f"Automatic main.py generation not supported for '{self.name}'.\n"
                f"Currently only 'ltx2' is supported.\n"
                f"For other examples, create a main.py manually in the deployment directory\n"
                f"with FastAPI endpoints for /health, /ready, and /run."
            )

        main_content = '''"""
Cerebrium FastAPI entry point for ltx2
Auto-generated by cvl deploy

Endpoints:
- GET /health - health check
- GET /ready - readiness check
- POST /run - generate video (T2V or I2V)

Supports:
- T2V (text-to-video): provide prompt only
- I2V (image-to-video): provide prompt + image_base64
"""

import os
import base64
import tempfile
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LTX-2 Video Generation")

# Global pipeline instance (loaded once on container start)
_pipeline = None
_model_paths = None


class RunRequest(BaseModel):
    """Request model for video generation."""
    prompt: str
    image_base64: Optional[str] = None
    height: int = 512
    width: int = 768
    num_frames: int = 33
    frame_rate: float = 24.0
    seed: int = 42


class RunResponse(BaseModel):
    """Response model for video generation."""
    video_base64: str
    format: str
    width: int
    height: int
    num_frames: int
    frame_rate: float
    mode: str


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    """Readiness check endpoint."""
    return {"status": "ready"}


def _ensure_pipeline():
    """Lazy-load the pipeline on first request."""
    global _pipeline, _model_paths

    if _pipeline is not None:
        return

    # Import here to avoid loading at module import time
    from predict import get_model_paths

    print("Loading LTX-2 models (first request, this takes ~60s)...")
    _model_paths = get_model_paths(pipeline="distilled", fp8=True)

    from ltx_pipelines.distilled import DistilledPipeline
    _pipeline = DistilledPipeline(
        checkpoint_path=_model_paths["checkpoint"],
        gemma_root=_model_paths["gemma"],
        spatial_upsampler_path=_model_paths["spatial_upsampler"],
        loras=[],
        fp8transformer=True,
    )
    print("Pipeline loaded!")


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest):
    """
    Generate video from text prompt, optionally with an input image (I2V).

    Args:
        request: RunRequest with prompt and optional parameters

    Returns:
        RunResponse with video_base64 containing the MP4 video
    """
    import torch
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

    _ensure_pipeline()

    # Create temp output file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    # Handle image input for I2V
    image_path = None
    images = []
    if request.image_base64:
        # Decode base64 image to temp file
        image_bytes = base64.b64decode(request.image_base64)
        # Detect format from magic bytes
        suffix = ".png" if image_bytes[:8] == b\'\\x89PNG\\r\\n\\x1a\\n\' else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_bytes)
            image_path = f.name
        images = [(image_path, 0, 1.0)]  # Image at frame 0, strength 1.0

    try:
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)

        with torch.inference_mode():
            video, audio = _pipeline(
                prompt=request.prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                frame_rate=request.frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=None,
                video_conditionings_stage1=[],
                video_conditionings_stage2=[],
            )

            encode_video(
                video=video,
                fps=request.frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        # Read and encode as base64
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return RunResponse(
            video_base64=base64.b64encode(video_bytes).decode("utf-8"),
            format="mp4",
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            frame_rate=request.frame_rate,
            mode="i2v" if request.image_base64 else "t2v",
        )

    finally:
        # Cleanup temp files
        if os.path.exists(output_path):
            os.remove(output_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
'''
        main_path = deploy_dir / "main.py"
        main_path.write_text(main_content)
        return str(main_path)

    def generate_dockerfile(self, deploy_dir: Path) -> str:
        """Generate Dockerfile for Cerebrium custom runtime.

        Closely follows the example's Dockerfile structure to maximize
        Docker layer cache reuse (same base, apt, pip steps).
        Only adds FastAPI/uvicorn and Cerebrium-specific settings at the end.
        """
        # Check if example has a Dockerfile to use as base
        example_dockerfile = self.example_path / "Dockerfile"
        base_image = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
        apt_packages = ["build-essential", "git", "curl", "ffmpeg", "libsndfile1"]

        if example_dockerfile.exists():
            parsed_base, parsed_apt = self._parse_dockerfile()
            if parsed_base:
                base_image = parsed_base
            if parsed_apt:
                apt_packages = parsed_apt

        # Build Dockerfile - structure matches original ltx2 Dockerfile for layer reuse
        dockerfile_lines = [
            "# Cerebrium deployment Dockerfile",
            "# Structure matches example Dockerfile for layer cache reuse",
            "",
            f"FROM {base_image}",
            "",
            "ENV DEBIAN_FRONTEND=noninteractive",
            "ENV PYTHONUNBUFFERED=1",
            "",
        ]

        # Add apt packages - same order/format as original
        if apt_packages:
            dockerfile_lines.extend([
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            ])
            # Format each package on its own line with backslash continuation
            for i, pkg in enumerate(apt_packages):
                if i < len(apt_packages) - 1:
                    dockerfile_lines.append(f"    {pkg} \\")
                else:
                    dockerfile_lines.append(f"    {pkg} \\")
            dockerfile_lines.extend([
                "    && rm -rf /var/lib/apt/lists/*",
                "",
            ])

        # Match original structure: /workspace, /tmp/requirements.txt
        dockerfile_lines.extend([
            "WORKDIR /workspace",
            "",
            "RUN pip install --upgrade pip setuptools wheel",
            "",
            "# Copy and install requirements (same path as original for layer reuse)",
            "COPY requirements.txt /tmp/requirements.txt",
            "RUN pip install --no-cache-dir -r /tmp/requirements.txt",
            "",
            "# Reinstall torchaudio to ensure it matches torch version",
            "RUN pip install --no-cache-dir torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
            "",
            "# --- Cerebrium additions below this line ---",
            "",
            "# Install FastAPI/uvicorn for Cerebrium custom runtime",
            "RUN pip install --no-cache-dir fastapi uvicorn",
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Add vendor directory to Python path",
            "ENV PYTHONPATH=/workspace/vendor",
            "",
            "# Use Cerebrium persistent storage for HuggingFace cache",
            "# Models uploaded during deploy are available here",
            "ENV HF_HOME=/persistent-storage/.cache/huggingface",
            "",
            "# PyTorch memory optimization",
            "ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "",
            "# Cerebrium requires EXPOSE and CMD for custom runtime",
            "EXPOSE 8192",
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]',
        ])

        dockerfile_content = "\n".join(dockerfile_lines) + "\n"
        dockerfile_path = deploy_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        return str(dockerfile_path)

    def prepare_deployment(self, deploy_dir: Optional[Path] = None) -> Path:
        """
        Prepare deployment directory with all required files.

        Args:
            deploy_dir: Optional directory to use (default: creates temp dir)

        Returns:
            Path to deployment directory
        """
        if deploy_dir is None:
            deploy_dir = Path(tempfile.mkdtemp(prefix=f"cvl-deploy-{self.name}-"))

        deploy_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing deployment in: {deploy_dir}")

        # Copy example code (exclude Dockerfile - we generate our own)
        print("  Copying example code...")
        for item in self.example_path.iterdir():
            if item.name.startswith("."):
                continue
            # Exclude files we'll generate or don't need
            if item.name in ("outputs", "__pycache__", "build.sh", "predict.sh", "test.sh", "Dockerfile"):
                continue
            dest = deploy_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        # Generate cerebrium.toml
        print("  Generating cerebrium.toml...")
        self.generate_cerebrium_toml(deploy_dir)

        # Generate Dockerfile for custom runtime
        print("  Generating Dockerfile...")
        self.generate_dockerfile(deploy_dir)

        # Generate main.py (FastAPI server)
        print("  Generating main.py (FastAPI server)...")
        self.generate_main_py(deploy_dir)

        return deploy_dir

    def deploy(self, deploy_dir: Path, dry_run: bool = False) -> int:
        """
        Run cerebrium deploy.

        Args:
            deploy_dir: Directory containing deployment files
            dry_run: If True, just print what would be done

        Returns:
            Exit code (0 for success)
        """
        if dry_run:
            print(f"\n[Dry run] Would deploy from: {deploy_dir}")
            print(f"[Dry run] Files:")
            for f in deploy_dir.iterdir():
                print(f"  - {f.name}")
            return 0

        print(f"\nDeploying to Cerebrium...")
        print(f"  Directory: {deploy_dir}")

        # Run cerebrium deploy
        result = subprocess.run(
            ["cerebrium", "deploy", "-y"],
            cwd=deploy_dir,
            text=True,
        )

        return result.returncode
