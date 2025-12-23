"""Path resolution utilities for CVL dual-mode execution.

This module provides path helpers for examples to work in both:
1. Standalone mode: bash predict.sh (workspace-relative paths)
2. CVL docker mode: cvl run --inputs/--outputs (user-specified directories)

Examples can import these utilities to handle input/output path resolution
and parameter loading across both execution modes.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict


def get_input_dir() -> Path:
    """Get input directory based on CVL execution context.

    Returns:
        Path to inputs directory:
        - CVL mode: /mnt/cvl/inputs (set by cvl run --inputs)
        - Standalone: ./inputs (relative to workspace)
    """
    return Path(os.getenv("CVL_INPUTS", "./inputs")).expanduser()


def get_output_dir() -> Path:
    """Get output directory based on CVL execution context.

    Creates the directory if it doesn't exist.

    Returns:
        Path to outputs directory:
        - CVL mode: /mnt/cvl/outputs (set by cvl run --outputs)
        - Standalone: ./outputs (relative to workspace)
    """
    out = Path(os.getenv("CVL_OUTPUTS", "./outputs")).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_input_path(path: str, input_dir: Optional[Path] = None) -> str:
    """Resolve input file path for dual-mode execution.

    Args:
        path: User-provided path (can be URL, absolute, or relative)
        input_dir: Base directory for relative paths (defaults to get_input_dir())

    Returns:
        Resolved absolute path or original URL

    Examples:
        # URLs and absolute paths pass through
        resolve_input_path("https://example.com/image.jpg")  # → unchanged
        resolve_input_path("/tmp/data.jpg")  # → unchanged

        # Relative paths resolve against input_dir
        # CVL mode: resolve_input_path("sample.jpg") → /mnt/cvl/inputs/sample.jpg
        # Standalone: resolve_input_path("sample.jpg") → ./inputs/sample.jpg
    """
    if input_dir is None:
        input_dir = get_input_dir()

    # URLs or absolute paths: use as-is
    if path.startswith(("http://", "https://")) or path.startswith("/"):
        return path

    # Relative path: resolve under input_dir if CVL_INPUTS set
    # Otherwise use as-is (for backward compatibility with standalone scripts)
    return str((input_dir / path).resolve()) if os.getenv("CVL_INPUTS") else path


def resolve_output_path(
    path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    default_filename: str = "result.txt"
) -> str:
    """Resolve output file path for dual-mode execution.

    Args:
        path: User-provided output path (can be absolute or relative)
        output_dir: Base directory for relative paths (defaults to get_output_dir())
        default_filename: Fallback filename if path is None

    Returns:
        Resolved absolute path

    Examples:
        # Absolute paths pass through (with warning in container mode)
        resolve_output_path("/tmp/output.txt")  # → /tmp/output.txt

        # Relative paths resolve against output_dir
        # CVL mode: resolve_output_path("result.txt") → /mnt/cvl/outputs/result.txt
        # Standalone: resolve_output_path("result.txt") → ./outputs/result.txt

        # None uses default
        resolve_output_path(None, default_filename="output.json")  # → .../output.json
    """
    if output_dir is None:
        output_dir = get_output_dir()

    # Use default filename if path not provided
    path = path or default_filename

    # Warn about absolute paths when CVL env vars are set - indicates we're
    # running via cvl run where absolute paths write to container filesystem
    # (not mounted to host), so files will be lost on exit
    if path.startswith("/") and _has_cvl_path_env():
        print(
            f"WARNING: Output path '{path}' is absolute and writes to container "
            f"filesystem. File may not be accessible on host. Use a relative path "
            f"to write to the mounted workspace instead.",
            file=sys.stderr
        )

    # Absolute paths stay absolute; relative paths resolve under output_dir
    return path if path.startswith("/") else str((output_dir / path).resolve())


def _has_cvl_path_env() -> bool:
    """Check if CVL path environment variables are set.

    Returns True if CVL_INPUTS or CVL_OUTPUTS is set, indicating paths
    should be resolved relative to CVL-mounted directories.
    """
    return bool(os.getenv("CVL_INPUTS") or os.getenv("CVL_OUTPUTS"))


def load_cvl_params() -> Dict[str, str]:
    """Load parameters from CVL environment and config file.

    Loads parameters from two sources (config.json takes precedence):
    1. Environment variables: CVL_PARAM_KEY=value → {"key": "value"}
    2. config.json in input directory (if exists)

    Returns:
        Dictionary of parameter key-value pairs

    Examples:
        # Via environment
        # CVL_PARAM_BATCH_SIZE=32 → {"batch_size": "32"}

        # Via config file
        # inputs/config.json: {"epochs": 10, "lr": 0.001}

        params = load_cvl_params()
        batch_size = int(params.get("batch_size", 16))
    """
    # Load from CVL_PARAM_* environment variables
    params = {
        k[10:].lower(): v
        for k, v in os.environ.items()
        if k.startswith("CVL_PARAM_")
    }

    # Load from config.json (if exists)
    input_dir = get_input_dir()
    cfg = input_dir / "config.json"
    if cfg.is_file():
        try:
            with cfg.open() as f:
                params.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            # Silently ignore malformed config
            pass

    return params


def setup_cvl_paths(default_output_filename: str = "result.txt") -> tuple[Path, Path]:
    """Convenience function to set up input/output directories.

    Args:
        default_output_filename: Default output filename (unused but kept for API)

    Returns:
        Tuple of (input_dir, output_dir)

    Example:
        INP, OUT = setup_cvl_paths()
        image_path = resolve_input_path(args.image, INP)
        output_path = resolve_output_path(args.output, OUT)
    """
    return get_input_dir(), get_output_dir()
