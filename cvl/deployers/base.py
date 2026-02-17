"""Base deployer with shared logic for all serverless platforms."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any


# Examples with automatic deployment support
SUPPORTED_EXAMPLES = {"ltx2"}

# Models required for each example (HuggingFace repo IDs and optional file list)
# Format: repo_id -> list of files (None means full repo)
EXAMPLE_MODELS = {
    "ltx2": {
        "Lightricks/LTX-2": [
            "ltx-2-19b-distilled-fp8.safetensors",
            "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        ],
        "google/gemma-3-12b-it-qat-q4_0-unquantized": None,
    },
}


class BaseDeployer:
    """Base class for serverless platform deployers."""

    platform_name: str = ""       # "Cerebrium", "Modal"
    dry_run_command: str = ""      # "cerebrium deploy", "modal deploy app.py"
    GPU_TABLE: dict = {}           # Override in subclass
    VRAM_TO_GPU: dict = {}         # Override in subclass

    def __init__(
        self,
        example_path: Path,
        example_meta: Dict[str, Any],
        gpu_override: Optional[str] = None,
    ):
        self.example_path = example_path
        self.example_meta = example_meta
        self.name = example_meta.get("name", example_path.name)
        self.gpu_override = gpu_override

    def is_supported(self) -> bool:
        return self.name in SUPPORTED_EXAMPLES

    def get_unsupported_message(self) -> str:
        return (
            f"Automatic {self.platform_name} deployment not yet supported for '{self.name}'.\n"
            f"Currently supported: {', '.join(sorted(SUPPORTED_EXAMPLES))}"
        )

    def get_gpu_config(self) -> tuple[str, str, int]:
        """Get (gpu_identifier, gpu_short_name, vram_gb)."""
        resources = self.example_meta.get("resources", {})
        vram_gb = resources.get("vram_gb", 24)

        if self.gpu_override and self.gpu_override in self.GPU_TABLE:
            gpu_info = self.GPU_TABLE[self.gpu_override]
            return gpu_info["id"], self.gpu_override, vram_gb
        elif self.gpu_override:
            return self.gpu_override, self.gpu_override, vram_gb

        for threshold, gpu_name in sorted(self.VRAM_TO_GPU.items()):
            if vram_gb <= threshold:
                return self.GPU_TABLE[gpu_name]["id"], gpu_name, vram_gb

        # Default to largest GPU
        last_gpu = list(self.VRAM_TO_GPU.values())[-1] if self.VRAM_TO_GPU else "H100"
        return self.GPU_TABLE.get(last_gpu, {}).get("id", last_gpu), last_gpu, vram_gb

    def format_gpu_info(self, gpu_name: str, gpu_id: str, vram_needed: int) -> list[str]:
        """Return extra GPU info lines for display. Override for platform-specific info."""
        gpu_info = self.GPU_TABLE.get(gpu_name, {})
        return [f"  VRAM: {gpu_info.get('vram', 'N/A')} GB"]

    def get_required_models(self) -> dict[str, list[str] | None]:
        return EXAMPLE_MODELS.get(self.name, {})

    # --- Shared utilities ---

    def _get_hf_cache_path(self, repo_id: str) -> Path:
        cache_name = "models--" + repo_id.replace("/", "--")
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        return Path(hf_home) / "hub" / cache_name

    def _download_model(self, repo_id: str, files: list[str] | None = None) -> Path:
        """Download model (or specific files) to local HuggingFace cache."""
        local_path = self._get_hf_cache_path(repo_id)

        if files is None:
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
            print(f"    Downloading {len(files)} files from {repo_id}...")
            try:
                from huggingface_hub import hf_hub_download
                for filename in files:
                    file_path = local_path / "snapshots"
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

    def _get_latest_snapshot(self, local_path: Path) -> tuple[Path, str] | None:
        """Find latest snapshot revision from HF cache.

        Returns (snapshot_path, revision_hash) or None.
        """
        snapshots_dir = local_path / "snapshots"
        if not snapshots_dir.exists():
            print(f"    Warning: No snapshots directory found")
            return None

        revisions = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not revisions:
            print(f"    Warning: No snapshot revisions found")
            return None

        return revisions[0], revisions[0].name

    def _resolve_snapshot_files(
        self, snapshot_path: Path, files: list[str] | None
    ) -> list[tuple[Path, str]] | None:
        """Resolve files in a snapshot to (real_path, filename) pairs.

        If files is None, resolves all files in the snapshot.
        Returns None on error.
        """
        if files:
            target_files = files
        else:
            target_files = [f.name for f in snapshot_path.iterdir()]

        resolved = []
        for filename in target_files:
            file_path = snapshot_path / filename
            if not file_path.exists() and not file_path.is_symlink():
                print(f"    Warning: File not found in snapshot: {filename}")
                return None

            real_path = file_path.resolve()
            if not real_path.exists():
                print(f"    Warning: Resolved file not found: {real_path}")
                return None

            resolved.append((real_path, filename))

        return resolved

    def _parse_dockerfile(self) -> tuple[Optional[str], list[str]]:
        """Parse example's Dockerfile to extract base image and apt packages."""
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
            while line.rstrip().endswith("\\") and i + 1 < len(lines):
                line = line.rstrip()[:-1] + " " + lines[i + 1].strip()
                i += 1
            joined_lines.append(line)
            i += 1

        for line in joined_lines:
            line = line.strip()
            if line.upper().startswith("FROM "):
                base_image = line.split()[1]
            if "apt-get install" in line:
                parts = line.split("apt-get install")[-1]
                for token in parts.split():
                    if token.startswith("-"):
                        continue
                    if token in ("&&", "||", ";", "rm", "apt-get", "update"):
                        break
                    if token and not token.startswith("/"):
                        apt_packages.append(token)

        return base_image, apt_packages

    def _get_base_image_and_packages(self) -> tuple[str, list[str]]:
        """Get base Docker image and apt packages from example Dockerfile."""
        base_image = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
        apt_packages = ["build-essential", "git", "curl", "ffmpeg", "libsndfile1"]

        if (self.example_path / "Dockerfile").exists():
            parsed_base, parsed_apt = self._parse_dockerfile()
            if parsed_base:
                base_image = parsed_base
            if parsed_apt:
                apt_packages = parsed_apt

        return base_image, apt_packages

    def _generate_base_dockerfile_lines(self, comment: str = "") -> list[str]:
        """Generate the common Dockerfile lines (FROM through pip install).

        Returns lines list. Platform deployers append their specific lines.
        """
        base_image, apt_packages = self._get_base_image_and_packages()

        lines = []
        if comment:
            lines.append(f"# {comment}")
            lines.append("")
        lines.extend([
            f"FROM {base_image}",
            "",
            "ENV DEBIAN_FRONTEND=noninteractive",
            "ENV PYTHONUNBUFFERED=1",
            "",
        ])

        if apt_packages:
            lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\")
            for pkg in apt_packages:
                lines.append(f"    {pkg} \\")
            lines.extend([
                "    && rm -rf /var/lib/apt/lists/*",
                "",
            ])

        lines.extend([
            "WORKDIR /workspace",
            "",
            "RUN pip install --upgrade pip setuptools wheel",
            "",
            "COPY requirements.txt /tmp/requirements.txt",
            "RUN pip install --no-cache-dir -r /tmp/requirements.txt",
            "",
            "# Reinstall torchaudio to ensure it matches torch version",
            "RUN pip install --no-cache-dir torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
            "",
        ])

        return lines

    def _copy_example_code(self, deploy_dir: Path) -> None:
        """Copy example code to deploy directory, excluding generated/build files."""
        print("  Copying example code...")
        for item in self.example_path.iterdir():
            if item.name.startswith("."):
                continue
            if item.name in ("outputs", "__pycache__", "build.sh", "predict.sh", "test.sh", "Dockerfile"):
                continue
            dest = deploy_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    # --- Model upload ---

    def _check_remote_storage(self, repo_id: str, files: list[str] | None = None) -> bool:
        """Check if model already exists in remote storage. Override for platforms that support it."""
        return False

    def _pre_upload(self) -> bool:
        """Called before upload loop. Override to create volumes, etc. Return False to abort."""
        return True

    def _upload_model_to_remote(self, repo_id: str, local_path: Path, files: list[str] | None = None) -> bool:
        """Upload a model to platform storage. Must be implemented by subclass."""
        raise NotImplementedError

    def upload_models(self) -> dict[str, bool]:
        """Download and upload required models to platform storage."""
        models = self.get_required_models()
        if not models:
            print("No models to upload for this example")
            return {}

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            print("\nError: huggingface_hub is required for model upload.")
            print("Install with: pip install huggingface_hub")
            print("\nAlternatively, use --skip-model-upload and upload models manually.")
            return {repo_id: False for repo_id in models}

        if not self._pre_upload():
            return {repo_id: False for repo_id in models}

        print(f"\nUploading models to {self.platform_name}...")
        results = {}

        for repo_id, files in models.items():
            if files:
                print(f"  Processing: {repo_id} ({len(files)} files)")
            else:
                print(f"  Processing: {repo_id} (full repo)")

            if self._check_remote_storage(repo_id, files):
                print(f"    Already in {self.platform_name} storage, skipping")
                results[repo_id] = True
                continue

            try:
                local_path = self._download_model(repo_id, files)
            except Exception:
                results[repo_id] = False
                continue

            success = self._upload_model_to_remote(repo_id, local_path, files)
            results[repo_id] = success

        uploaded = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        print(f"\nModel upload complete: {uploaded} succeeded, {failed} failed")

        return results

    # --- Abstract methods ---

    def check_cli(self) -> tuple[bool, str]:
        raise NotImplementedError

    def prepare_deployment(self, deploy_dir: Optional[Path] = None) -> Path:
        raise NotImplementedError

    def deploy(self, deploy_dir: Path, dry_run: bool = False) -> int:
        raise NotImplementedError
