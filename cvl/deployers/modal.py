"""Modal serverless deployment."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from .base import BaseDeployer


# Modal GPU options — Modal uses simple string identifiers
MODAL_GPUS = {
    "T4": {"id": "T4", "vram": 16},
    "L4": {"id": "L4", "vram": 24},
    "A10": {"id": "A10G", "vram": 24},
    "L40": {"id": "L40S", "vram": 48},
    "A100": {"id": "A100-40GB", "vram": 40},
    "A100_80GB": {"id": "A100-80GB", "vram": 80},
    "H100": {"id": "H100", "vram": 80},
}

# Modal volume name per example
EXAMPLE_VOLUME_NAMES = {
    "ltx2": "ltx2-models",
}


class ModalDeployer(BaseDeployer):
    """Deploy CVL examples to Modal serverless platform."""

    platform_name = "Modal"
    dry_run_command = "modal deploy app.py"
    GPU_TABLE = MODAL_GPUS
    VRAM_TO_GPU = {
        16: "T4",
        24: "A10",
        40: "L40",
        48: "L40",
        80: "H100",
    }

    def _get_volume_name(self) -> str:
        return EXAMPLE_VOLUME_NAMES.get(self.name, f"{self.name}-models")

    def check_cli(self) -> tuple[bool, str]:
        cli_path = shutil.which("modal")
        if not cli_path:
            return False, (
                "Modal CLI not found.\n"
                "Install with: pip install modal\n"
                "Then run: modal setup"
            )

        print(f"  Found: {cli_path}", flush=True)

        # Check authentication via modal profile current
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if not os.environ.get("MODAL_TOKEN_ID"):
                return False, (
                    "Not authenticated with Modal.\n"
                    "  Run: modal setup\n"
                    "  Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables"
                )

        return True, "Modal CLI ready"

    # --- Model upload (Modal-specific) ---

    def _pre_upload(self) -> bool:
        """Create Modal volume if it doesn't exist."""
        volume_name = self._get_volume_name()
        result = subprocess.run(
            ["modal", "volume", "list"],
            capture_output=True,
            text=True,
        )
        if volume_name in result.stdout:
            print(f"    Volume '{volume_name}' already exists")
            return True

        print(f"    Creating volume '{volume_name}'...")
        result = subprocess.run(
            ["modal", "volume", "create", volume_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Error creating volume: {result.stderr}")
            return False
        return True

    def _upload_model_to_remote(self, repo_id: str, local_path: Path, files: list[str] | None = None) -> bool:
        """Upload model from local HF cache to Modal volume."""
        volume_name = self._get_volume_name()
        cache_name = "models--" + repo_id.replace("/", "--")
        remote_base = f"hub/{cache_name}"

        print(f"    Uploading {repo_id} to Modal volume '{volume_name}'...")

        # Upload refs directory
        refs_dir = local_path / "refs"
        if refs_dir.exists():
            print(f"      Uploading refs...")
            result = subprocess.run(
                ["modal", "volume", "put", volume_name, str(refs_dir), f"{remote_base}/refs"],
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
                ["modal", "volume", "put", volume_name, str(real_path), remote_path],
                text=True,
            )
            if result.returncode != 0:
                print(f"    Warning: Upload failed for {filename}")
                return False

        print(f"    Uploaded: {repo_id}")
        return True

    # --- Deployment file generation ---

    def generate_dockerfile(self, deploy_dir: Path) -> str:
        lines = self._generate_base_dockerfile_lines(
            "Modal deployment Dockerfile — no CMD/EXPOSE needed"
        )
        lines.extend([
            "# Copy application code",
            "COPY . .",
            "",
            "ENV PYTHONPATH=/workspace/vendor",
            "",
            "ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ])

        dockerfile_path = deploy_dir / "Dockerfile"
        dockerfile_path.write_text("\n".join(lines) + "\n")
        return str(dockerfile_path)

    def generate_app_py(self, deploy_dir: Path) -> str:
        if self.name != "ltx2":
            raise NotImplementedError(
                f"Automatic app.py generation not supported for '{self.name}'.\n"
                f"Currently only 'ltx2' is supported."
            )

        gpu_id, _, _ = self.get_gpu_config()
        volume_name = self._get_volume_name()

        app_content = f'''"""
Modal app definition for ltx2
Auto-generated by cvl deploy
"""

import modal

app = modal.App("{self.name}")

image = modal.Image.from_dockerfile("./Dockerfile")

vol = modal.Volume.from_name("{volume_name}", create_if_missing=True)


@app.cls(
    image=image,
    gpu="{gpu_id}",
    volumes={{"/root/.cache/huggingface": vol}},
    timeout=600,
    scaledown_window=300,
    min_containers=0,
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        """Load model once per container start."""
        import torch  # noqa: F401
        from predict import get_model_paths
        from ltx_pipelines.distilled import DistilledPipeline

        print("Loading LTX-2 models...")
        self.model_paths = get_model_paths(pipeline="distilled", fp8=True)
        self.pipeline = DistilledPipeline(
            checkpoint_path=self.model_paths["checkpoint"],
            gemma_root=self.model_paths["gemma"],
            spatial_upsampler_path=self.model_paths["spatial_upsampler"],
            loras=[],
            fp8transformer=True,
        )
        print("Pipeline loaded!")

    @modal.asgi_app()
    def serve(self):
        import os
        import base64
        import tempfile
        from typing import Optional

        from fastapi import FastAPI
        from pydantic import BaseModel

        fastapi_app = FastAPI(title="LTX-2 Video Generation")

        class RunRequest(BaseModel):
            prompt: str
            image_base64: Optional[str] = None
            height: int = 512
            width: int = 768
            num_frames: int = 33
            frame_rate: float = 24.0
            seed: int = 42
            stage1_steps: Optional[int] = None
            stage2_steps: Optional[int] = None

        class RunResponse(BaseModel):
            video_base64: str
            format: str
            width: int
            height: int
            num_frames: int
            frame_rate: float
            mode: str

        pipeline = self.pipeline

        @fastapi_app.get("/health")
        def health():
            return {{"status": "healthy"}}

        @fastapi_app.get("/ready")
        def ready():
            return {{"status": "ready"}}

        @fastapi_app.post("/run", response_model=RunResponse)
        def run(request: RunRequest):
            import torch
            from ltx_pipelines.utils.media_io import encode_video
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                output_path = f.name

            image_path = None
            images = []
            if request.image_base64:
                image_bytes = base64.b64decode(request.image_base64)
                suffix = ".png" if image_bytes[:8] == b\'\\x89PNG\\r\\n\\x1a\\n\' else ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(image_bytes)
                    image_path = f.name
                images = [(image_path, 0, 1.0)]

            try:
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)

                from predict import get_sigma_schedule, DEFAULT_STAGE1_SIGMAS, DEFAULT_STAGE2_SIGMAS
                stage1_sigmas = get_sigma_schedule(request.stage1_steps, DEFAULT_STAGE1_SIGMAS)
                stage2_sigmas = get_sigma_schedule(request.stage2_steps, DEFAULT_STAGE2_SIGMAS)

                with torch.inference_mode():
                    video, audio = pipeline(
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
                if os.path.exists(output_path):
                    os.remove(output_path)
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)

        return fastapi_app
'''
        app_path = deploy_dir / "app.py"
        app_path.write_text(app_content)
        return str(app_path)

    def prepare_deployment(self, deploy_dir: Optional[Path] = None) -> Path:
        if deploy_dir is None:
            deploy_dir = Path(tempfile.mkdtemp(prefix=f"cvl-deploy-{self.name}-modal-"))
        deploy_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing deployment in: {deploy_dir}")
        self._copy_example_code(deploy_dir)

        print("  Generating Dockerfile...")
        self.generate_dockerfile(deploy_dir)

        print("  Generating app.py (Modal app)...")
        self.generate_app_py(deploy_dir)

        return deploy_dir

    def deploy(self, deploy_dir: Path, dry_run: bool = False) -> int:
        if dry_run:
            print(f"\n[Dry run] Would deploy from: {deploy_dir}")
            return 0

        print(f"\nDeploying to Modal...")
        print(f"  Directory: {deploy_dir}")

        result = subprocess.run(
            ["modal", "deploy", "app.py"],
            cwd=deploy_dir,
            text=True,
        )
        return result.returncode
