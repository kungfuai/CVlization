"""Export command - bundle an example with the cvlization package."""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cvl.core.discovery import find_repo_root
from cvl.core.matching import find_matching_examples


IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".DS_Store",
    ".git",
]


def _copytree(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Copy a directory tree with a small ignore list."""
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination '{dst}' already exists.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=overwrite,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
    )


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _needs_cvlization(src_dir: Path) -> bool:
    """Check if any .py file under src_dir imports from cvlization."""
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            if "import cvlization" in content or "from cvlization" in content:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Template content for Potluck export
# ---------------------------------------------------------------------------


def _compose_yaml_content(image_name: str) -> str:
    return textwrap.dedent(f"""\
        services:
          app:
            build:
              context: .
              dockerfile: services/app/Dockerfile
            image: {image_name}
            volumes:
              - .:/workspace
              - ${{HF_CACHE}}:/root/.cache/huggingface
              - ${{CVL_CACHE}}:/root/.cache/cvlization
            environment:
              - PYTHONPATH=/workspace
              - PYTHONUNBUFFERED=1
              - HF_TOKEN=${{HF_TOKEN:-}}
              - WANDB_API_KEY=${{WANDB_API_KEY:-}}
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: all
                      capabilities: [gpu]
            shm_size: '16g'
            ipc: host
    """)


def _setup_environment_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        # setup_environment.sh -- GPU detection, .env sourcing, cache dir export.
        set -euo pipefail

        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

        # Source .env if present
        if [ -f "$PROJECT_ROOT/.env" ]; then
          set -a
          source "$PROJECT_ROOT/.env"
          set +a
        fi

        # Cache directories
        export HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
        export CVL_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/cvlization"
        mkdir -p "$HF_CACHE" "$CVL_CACHE"

        # GPU detection
        if command -v nvidia-smi &>/dev/null; then
          echo "[setup] NVIDIA GPU detected"
          nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        else
          echo "[setup] WARNING: nvidia-smi not found -- GPU may not be available"
        fi
    """)


def _bin_build_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        source "$(dirname "${BASH_SOURCE[0]}")/setup_environment.sh"
        cd "$PROJECT_ROOT"
        docker compose build
    """)


def _bin_train_sh() -> str:
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        source "$(dirname "${BASH_SOURCE[0]}")/setup_environment.sh"
        cd "$PROJECT_ROOT"
        docker compose run --rm app python -m src.training.training_session "$@"
    """)


def _env_example() -> str:
    return textwrap.dedent("""\
        # Rename to .env and fill in values
        HF_TOKEN=
        WANDB_API_KEY=
    """)


def _potluck_export_readme(example_name: str, commit: Optional[str]) -> str:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return textwrap.dedent(f"""\
        # {example_name} (Exported)

        - Exported at: {now}
        - Source commit: {commit or 'unknown'}

        ## Quick start

        ```bash
        # 1. Copy .env.example to .env and fill in tokens
        cp .env.example .env

        # 2. Build the Docker image
        bash bin/build.sh

        # 3. Run training
        bash bin/train.sh --epochs 2
        ```

        ## Structure

        - `src/` -- Python training code (Potluck interfaces)
        - `services/app/Dockerfile` -- Docker build definition
        - `compose.yaml` -- Docker Compose orchestration
        - `bin/` -- Entry-point scripts (build, train, setup)
        - `requirements.txt` -- Python dependencies

        ## Notes

        - GPU access requires the NVIDIA Container Toolkit.
        - HuggingFace and Weights & Biases tokens are read from `.env`.
    """)


# ---------------------------------------------------------------------------
# Legacy export (non-Potluck examples)
# ---------------------------------------------------------------------------


def _write_legacy_readme(
    export_root: Path, rel_example_path: Path, commit: Optional[str]
) -> None:
    """Drop a small helper README into the export."""
    readme = export_root / "EXPORT_README.md"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    levels_up = "/".join([".."] * len(rel_example_path.parts))

    content = [
        "# CVlization Export",
        "",
        f"- Example: {rel_example_path.as_posix()}",
        f"- Exported at: {now}",
        f"- Source commit: {commit or 'unknown'}",
        "",
        "## Structure",
        "- `examples/...` contains your copied example",
        "- `cvlization/` is a vendored snapshot so imports work without the repo",
        "",
        "## Run the example",
        "From the example directory:",
        "```bash",
        f"cd {rel_example_path.as_posix()}",
        "bash build.sh    # if the example provides a Docker build",
        "bash train.sh    # or predict.sh, etc.",
        "```",
        "",
        "If you run Python directly instead of via the scripts, set PYTHONPATH to the",
        "export root so the bundled `cvlization` is visible:",
        "```bash",
        f"EXPORT_ROOT=\"$(cd \"$(dirname \"$0\")/{levels_up}\" && pwd)\"",
        "PYTHONPATH=\"$EXPORT_ROOT\" python3 train.py",
        "```",
        "",
        "## Notes",
        "- The export keeps the `examples/...` path so existing scripts that compute",
        "  the repo root by walking up directories still work.",
        "- Dependencies beyond `cvlization` are still managed by the example's own",
        "  Dockerfile/requirements files.",
    ]

    readme.write_text("\n".join(content))


def _export_legacy(
    example_src: Path,
    rel_path: Path,
    repo_root: Path,
    export_root: Path,
    overwrite: bool,
) -> Tuple[int, str]:
    """Legacy export: copy example + cvlization package as-is."""
    try:
        _copytree(example_src, export_root / rel_path, overwrite=overwrite)
        _copytree(
            repo_root / "cvlization",
            export_root / "cvlization",
            overwrite=overwrite,
        )
    except Exception as e:
        return 1, f"Failed to copy files: {e}"

    _write_legacy_readme(export_root, rel_path, _git_commit(repo_root))
    return 0, f"Exported to {export_root}"


# ---------------------------------------------------------------------------
# Potluck export
# ---------------------------------------------------------------------------


def _export_potluck(
    example_src: Path,
    example_meta: Dict,
    repo_root: Path,
    export_root: Path,
    overwrite: bool,
) -> Tuple[int, str]:
    """Export a Potluck-structured example with generated scaffolding."""
    try:
        # 1. Copy src/ as-is
        _copytree(example_src / "src", export_root / "src", overwrite=overwrite)

        # 2. Copy requirements.txt to root
        req_src = example_src / "requirements.txt"
        if req_src.exists():
            shutil.copy2(req_src, export_root / "requirements.txt")

        # 3. Copy Dockerfile -> services/app/Dockerfile, patching COPY path
        dockerfile_src = example_src / "Dockerfile"
        if dockerfile_src.exists():
            services_dir = export_root / "services" / "app"
            services_dir.mkdir(parents=True, exist_ok=True)
            dockerfile_content = dockerfile_src.read_text()
            dockerfile_content = dockerfile_content.replace(
                "COPY requirements.txt",
                "COPY services/app/requirements.txt",
            )
            (services_dir / "Dockerfile").write_text(dockerfile_content)
            # Copy requirements.txt alongside Dockerfile for the patched COPY
            if req_src.exists():
                shutil.copy2(req_src, services_dir / "requirements.txt")

        # 4. Generate compose.yaml
        image_name = example_meta.get("image", example_meta.get("name", "app"))
        (export_root / "compose.yaml").write_text(
            _compose_yaml_content(image_name)
        )

        # 5. Generate bin/ scripts
        bin_dir = export_root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        setup_path = bin_dir / "setup_environment.sh"
        setup_path.write_text(_setup_environment_sh())
        setup_path.chmod(0o755)

        build_path = bin_dir / "build.sh"
        build_path.write_text(_bin_build_sh())
        build_path.chmod(0o755)

        train_path = bin_dir / "train.sh"
        train_path.write_text(_bin_train_sh())
        train_path.chmod(0o755)

        # 6. Generate .env.example
        (export_root / ".env.example").write_text(_env_example())

        # 7. Check if cvlization imports are needed; if so, copy package
        if _needs_cvlization(export_root / "src"):
            cvl_src = repo_root / "cvlization"
            if cvl_src.is_dir():
                _copytree(
                    cvl_src,
                    export_root / "cvlization",
                    overwrite=overwrite,
                )

        # 8. Generate EXPORT_README.md
        example_name = example_meta.get("name", example_src.name)
        (export_root / "EXPORT_README.md").write_text(
            _potluck_export_readme(example_name, _git_commit(repo_root))
        )

    except Exception as e:
        return 1, f"Failed to export: {e}"

    return 0, f"Exported to {export_root}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def export_example(
    examples: List[Dict],
    example_identifier: str,
    dest: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[int, str]:
    """Export an example directory plus the cvlization package.

    For Potluck-compatible examples (those with a ``src/`` directory), generates
    Docker Compose scaffolding (compose.yaml, bin/, services/).  For legacy
    examples, copies the example tree alongside a vendored ``cvlization/``
    package.

    Returns:
        (exit_code, message). exit_code==0 indicates success.
    """
    matches, suggestions = find_matching_examples(examples, example_identifier)
    if not matches:
        hint = ""
        if suggestions:
            hint = f" Suggestions: {', '.join(suggestions)}"
        return 1, f"Example '{example_identifier}' not found.{hint}"
    if len(matches) > 1:
        suggested = ", ".join(
            m.get("_path", "") for m in matches if m.get("_path")
        )
        return 1, f"Ambiguous example '{example_identifier}'. Matches: {suggested}"

    repo_root = find_repo_root()
    rel_path = Path(matches[0]["_path"])
    example_src = repo_root / rel_path
    if not example_src.exists():
        return 1, f"Example path does not exist: {example_src}"

    export_root = Path(dest) if dest else Path.cwd() / f"{rel_path.name}-export"
    export_root = export_root.resolve()

    if export_root.exists() and not overwrite and any(export_root.iterdir()):
        return (
            1,
            f"Destination '{export_root}' already exists and is not empty. "
            "Use --force to overwrite.",
        )
    export_root.mkdir(parents=True, exist_ok=True)

    # Detect Potluck-structured example (has src/ directory)
    is_potluck_compatible = (example_src / "src").is_dir()

    if is_potluck_compatible:
        return _export_potluck(
            example_src, matches[0], repo_root, export_root, overwrite
        )
    else:
        return _export_legacy(
            example_src, rel_path, repo_root, export_root, overwrite
        )
