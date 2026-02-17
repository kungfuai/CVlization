"""Cerebrium serverless deployment."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from .base import BaseDeployer


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

# Cerebrium persistent storage path for HuggingFace cache
CEREBRIUM_HF_CACHE = ".cache/huggingface/hub"


class CerebriumDeployer(BaseDeployer):
    """Deploy CVL examples to Cerebrium serverless platform."""

    platform_name = "Cerebrium"
    dry_run_command = "cerebrium deploy"
    GPU_TABLE = CEREBRIUM_GPUS
    VRAM_TO_GPU = {
        16: "T4",
        24: "A10",
        40: "L40",
        48: "L40",
        80: "A100_80GB",
    }

    def __init__(
        self,
        example_path: Path,
        example_meta: Dict[str, Any],
        gpu_override: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        super().__init__(example_path, example_meta, gpu_override)
        self.project_id = project_id

    def format_gpu_info(self, gpu_name: str, gpu_id: str, vram_needed: int) -> list[str]:
        gpu_info = self.GPU_TABLE.get(gpu_name, {})
        return [
            f"  VRAM: {gpu_info.get('vram', 'N/A')} GB",
            f"  Plan: {gpu_info.get('plan', 'N/A')}",
        ]

    def check_cli(self) -> tuple[bool, str]:
        cli_path = shutil.which("cerebrium")
        if not cli_path:
            return False, (
                "Cerebrium CLI not found.\n"
                "Install with: uv add cerebrium  (or: pip install cerebrium)\n"
                "Then run: cerebrium login"
            )

        print(f"  Found: {cli_path}", flush=True)

        config_path = Path.home() / ".cerebrium" / "config.yaml"
        if not config_path.exists():
            if not os.environ.get("CEREBRIUM_TOKEN"):
                return False, (
                    "Not logged in to Cerebrium.\n"
                    "  Run: cerebrium login\n"
                    "  Or set CEREBRIUM_TOKEN environment variable for CI/CD"
                )

        return True, "Cerebrium CLI ready"

    # --- Model upload (Cerebrium-specific) ---

    def _check_remote_storage(self, repo_id: str, files: list[str] | None = None) -> bool:
        """Check if model exists in Cerebrium persistent storage."""
        cache_name = "models--" + repo_id.replace("/", "--")
        snapshots_path = f"{CEREBRIUM_HF_CACHE}/{cache_name}/snapshots"

        result = subprocess.run(
            ["cerebrium", "ls", snapshots_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        if "No files found" in result.stdout or "No files found" in result.stderr:
            return False

        if files:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l and not l.startswith("NAME")]
            if not lines:
                return False

            revision_dir = lines[0].split()[0].rstrip("/")
            revision_path = f"{snapshots_path}/{revision_dir}"

            result = subprocess.run(
                ["cerebrium", "ls", revision_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or "No files found" in result.stdout:
                return False

            existing_files = result.stdout
            for filename in files:
                if filename not in existing_files:
                    return False

            if ".incomplete" in existing_files:
                return False

        return True

    def _upload_model_to_remote(self, repo_id: str, local_path: Path, files: list[str] | None = None) -> bool:
        """Upload model from local cache to Cerebrium persistent storage."""
        cache_name = "models--" + repo_id.replace("/", "--")
        remote_base = f"{CEREBRIUM_HF_CACHE}/{cache_name}"

        print(f"    Uploading {repo_id} to Cerebrium...")
        print(f"      Local: {local_path}")
        print(f"      Remote: /persistent-storage/{remote_base}")

        # Upload refs directory
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

        # Upload snapshot files
        snapshot_info = self._get_latest_snapshot(local_path)
        if not snapshot_info:
            return False

        latest_snapshot, revision = snapshot_info
        resolved = self._resolve_snapshot_files(latest_snapshot, files)
        if resolved is None:
            return False

        total_size = sum(p.stat().st_size for p, _ in resolved)
        print(f"      Uploading {len(resolved)} files ({total_size / 1e9:.1f}GB) from snapshot {revision[:8]}...")

        for real_path, filename in resolved:
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

        print(f"    Uploaded: {repo_id}")
        return True

    # --- Deployment file generation ---

    def _parse_requirements(self) -> dict[str, str]:
        """Parse requirements.txt to extract pip packages."""
        requirements = self.example_path / "requirements.txt"
        if not requirements.exists():
            return {}

        packages = {}
        for line in requirements.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            for op in ["==", ">=", "<=", "~=", "!="]:
                if op in line:
                    name, version = line.split(op, 1)
                    packages[name.strip()] = f"{op}{version.strip()}"
                    break
            else:
                packages[line] = "latest"
        return packages

    def generate_cerebrium_toml(self, deploy_dir: Path) -> str:
        resources = self.example_meta.get("resources", {})
        vram_gb = resources.get("vram_gb", 24)
        gpu_count = resources.get("gpu_count", resources.get("gpu", 1))
        gpu_id, _, _ = self.get_gpu_config()
        memory_gb = max(32, int(vram_gb * 1.5))

        toml_lines = [
            "[cerebrium.deployment]",
            f'name = "{self.name}"',
        ]
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
            "[cerebrium.runtime.custom]",
            "port = 8192",
            'healthcheck_endpoint = "/health"',
            'readycheck_endpoint = "/ready"',
            'dockerfile_path = "./Dockerfile"',
        ])

        toml_path = deploy_dir / "cerebrium.toml"
        toml_path.write_text("\n".join(toml_lines) + "\n")
        return str(toml_path)

    def generate_main_py(self, deploy_dir: Path) -> str:
        if self.name != "ltx2":
            raise NotImplementedError(
                f"Automatic main.py generation not supported for '{self.name}'.\n"
                f"Currently only 'ltx2' is supported."
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
    stage1_steps: Optional[int] = None  # EXPERIMENTAL: Override stage 1 steps (default 8)
    stage2_steps: Optional[int] = None  # EXPERIMENTAL: Override stage 2 steps (default 3)


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

        # Get sigma schedules (EXPERIMENTAL: custom step counts)
        from predict import get_sigma_schedule, DEFAULT_STAGE1_SIGMAS, DEFAULT_STAGE2_SIGMAS
        stage1_sigmas = get_sigma_schedule(request.stage1_steps, DEFAULT_STAGE1_SIGMAS)
        stage2_sigmas = get_sigma_schedule(request.stage2_steps, DEFAULT_STAGE2_SIGMAS)

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
                stage1_sigmas=stage1_sigmas,
                stage2_sigmas=stage2_sigmas,
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
        lines = self._generate_base_dockerfile_lines(
            "Cerebrium deployment Dockerfile"
        )
        lines.extend([
            "# Install FastAPI/uvicorn for Cerebrium custom runtime",
            "RUN pip install --no-cache-dir fastapi uvicorn",
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "ENV PYTHONPATH=/workspace/vendor",
            "",
            "# Use Cerebrium persistent storage for HuggingFace cache",
            "ENV HF_HOME=/persistent-storage/.cache/huggingface",
            "",
            "ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "",
            "EXPOSE 8192",
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]',
        ])

        dockerfile_path = deploy_dir / "Dockerfile"
        dockerfile_path.write_text("\n".join(lines) + "\n")
        return str(dockerfile_path)

    def prepare_deployment(self, deploy_dir: Optional[Path] = None) -> Path:
        if deploy_dir is None:
            deploy_dir = Path(tempfile.mkdtemp(prefix=f"cvl-deploy-{self.name}-"))
        deploy_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing deployment in: {deploy_dir}")
        self._copy_example_code(deploy_dir)

        print("  Generating cerebrium.toml...")
        self.generate_cerebrium_toml(deploy_dir)

        print("  Generating Dockerfile...")
        self.generate_dockerfile(deploy_dir)

        print("  Generating main.py (FastAPI server)...")
        self.generate_main_py(deploy_dir)

        return deploy_dir

    def deploy(self, deploy_dir: Path, dry_run: bool = False) -> int:
        if dry_run:
            print(f"\n[Dry run] Would deploy from: {deploy_dir}")
            print(f"[Dry run] Files:")
            for f in deploy_dir.iterdir():
                print(f"  - {f.name}")
            return 0

        print(f"\nDeploying to Cerebrium...")
        print(f"  Directory: {deploy_dir}")

        result = subprocess.run(
            ["cerebrium", "deploy", "-y"],
            cwd=deploy_dir,
            text=True,
        )
        return result.returncode
