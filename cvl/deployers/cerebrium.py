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
    ):
        """
        Initialize deployer.

        Args:
            example_path: Path to example directory
            example_meta: Example metadata from example.yaml
            gpu_override: Override GPU type (e.g., "A10", "A100", "H100")
        """
        self.example_path = example_path
        self.example_meta = example_meta
        self.name = example_meta.get("name", example_path.name)
        self.gpu_override = gpu_override

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
        """Generate cerebrium.toml configuration."""
        resources = self.example_meta.get("resources", {})
        vram_gb = resources.get("vram_gb", 24)
        gpu_count = resources.get("gpu_count", resources.get("gpu", 1))
        disk_gb = resources.get("disk_gb", 50)

        gpu_id, gpu_name, _ = self.get_gpu_config()

        # Estimate memory: vram + some overhead for CPU operations
        memory_gb = max(16, vram_gb // 2)

        # Parse Dockerfile for base image and apt packages
        base_image, apt_packages = self._parse_dockerfile()

        # Parse requirements.txt for pip packages
        pip_packages = self._parse_requirements()

        # Start building toml content
        toml_lines = [
            "[cerebrium.deployment]",
            f'name = "{self.name}"',
            'python_version = "3.11"',
            'include = ["*", "**/*"]',
            'exclude = [".*", "__pycache__", "*.pyc", "outputs"]',
        ]

        # Add base image if found in Dockerfile
        if base_image:
            toml_lines.append(f'docker_base_image_url = "{base_image}"')

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
        ])

        # Add apt packages if found
        if apt_packages:
            toml_lines.extend([
                "",
                "[cerebrium.dependencies.apt]",
            ])
            for pkg in apt_packages:
                toml_lines.append(f'{pkg} = "latest"')

        # Add pip packages
        toml_lines.extend([
            "",
            "[cerebrium.dependencies.pip]",
        ])

        # Use packages from requirements.txt if available, else defaults
        if pip_packages:
            for name, version in pip_packages.items():
                if version == "latest":
                    toml_lines.append(f'{name} = "latest"')
                else:
                    toml_lines.append(f'{name} = "{version}"')
        else:
            # Fallback defaults
            toml_lines.extend([
                'torch = ">=2.0.0"',
                'torchaudio = ">=2.0.0"',
                'transformers = ">=4.40.0"',
                'accelerate = ">=0.30.0"',
                'safetensors = ">=0.4.0"',
                'huggingface_hub = ">=0.23.0"',
            ])

        toml_content = "\n".join(toml_lines) + "\n"
        toml_path = deploy_dir / "cerebrium.toml"
        toml_path.write_text(toml_content)
        return str(toml_path)

    def generate_main_py(self, deploy_dir: Path) -> str:
        """Generate main.py entry point for Cerebrium.

        Currently only supports ltx2 example. Other examples will need
        manual main.py creation or future implementation.
        """
        if self.name != "ltx2":
            raise NotImplementedError(
                f"Automatic main.py generation not supported for '{self.name}'.\n"
                f"Currently only 'ltx2' is supported.\n"
                f"For other examples, create a main.py manually in the deployment directory\n"
                f"with a run() function that Cerebrium will call."
            )

        main_content = f'''"""
Cerebrium entry point for {self.name}
Auto-generated by cvl deploy
"""

import os
import sys
import base64
import tempfile

# Add example directory to path
sys.path.insert(0, "/app")

# Global pipeline instance (loaded once on container start)
_pipeline = None
_model_paths = None


def _ensure_pipeline():
    """Lazy-load the pipeline on first request."""
    global _pipeline, _model_paths

    if _pipeline is not None:
        return

    # Import here to avoid loading at module import time
    from predict import get_model_paths, DistilledPipeline

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


def run(
    prompt: str,
    height: int = 512,
    width: int = 768,
    num_frames: int = 33,
    frame_rate: float = 24.0,
    seed: int = 42,
) -> dict:
    """
    Generate video from text prompt.

    Args:
        prompt: Text description of the video to generate
        height: Video height (default 512)
        width: Video width (default 768)
        num_frames: Number of frames (default 33, ~1.4s at 24fps)
        frame_rate: Frame rate (default 24.0)
        seed: Random seed for reproducibility

    Returns:
        dict with "video_base64" containing the MP4 video encoded as base64
    """
    import torch
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

    _ensure_pipeline()

    # Create temp output file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    try:
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        with torch.inference_mode():
            video, audio = _pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[],
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=None,
                video_conditionings_stage1=[],
                video_conditionings_stage2=[],
            )

            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        # Read and encode as base64
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return {{
            "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
        }}

    finally:
        # Cleanup temp file
        if os.path.exists(output_path):
            os.remove(output_path)
'''
        main_path = deploy_dir / "main.py"
        main_path.write_text(main_content)
        return str(main_path)

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

        # Copy example code
        print("  Copying example code...")
        for item in self.example_path.iterdir():
            if item.name.startswith("."):
                continue
            if item.name in ("outputs", "__pycache__", "build.sh", "predict.sh", "test.sh"):
                continue
            dest = deploy_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        # Generate cerebrium.toml
        print("  Generating cerebrium.toml...")
        self.generate_cerebrium_toml(deploy_dir)

        # Generate main.py
        print("  Generating main.py...")
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
