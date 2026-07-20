#!/usr/bin/env python3
"""KnowAct-GUIClaw: cross-platform GUI agent with self-evolving memory and skills.

Wraps the upstream GUIClaw CLI (https://github.com/HITsz-TMG/KnowAct) into a
CVlization-compatible inference example. Runs in dry-run mode by default to
demonstrate the Know-Route-Act-Reflect architecture without requiring a live
device. Connect an Android device/emulator via ADB for full GUI automation.

Requires an OpenAI-compatible VLM endpoint (e.g. vLLM serving Qwen3.5-35B).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from cvlization.paths import resolve_output_path, resolve_input_path
except ImportError:
    def resolve_input_path(path: str, input_dir: Optional[Path] = None) -> str:
        return path

    def resolve_output_path(
        path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        default_filename: str = "result.json",
    ) -> str:
        out = Path(os.environ.get("CVL_OUTPUTS", "."))
        out.mkdir(parents=True, exist_ok=True)
        p = path or default_filename
        return p if p.startswith("/") else str((out / p).resolve())


HF_DATA_REPO = "zzsi/cvl"
HF_DATA_PREFIX = "knowact_guiclaw"


def ensure_sample_data(cache_root: Optional[Path] = None) -> Path:
    """Download sample Android screenshot from HuggingFace if not cached."""
    if cache_root is None:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        cache_root = hf_home / "cvl_data" / HF_DATA_PREFIX

    marker = cache_root / ".downloaded"
    if marker.exists():
        print("Sample data already cached.", file=sys.stderr)
        return cache_root

    try:
        from huggingface_hub import hf_hub_download

        files = ["sample_screenshot.png"]
        for rel_path in files:
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                filename=f"{HF_DATA_PREFIX}/{rel_path}",
                repo_type="dataset",
            )
            local_target = cache_root / rel_path
            local_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, local_target)
            print(f"Downloaded {rel_path}", file=sys.stderr)

        marker.touch()
    except Exception as e:
        print(
            f"Warning: could not download sample data from {HF_DATA_REPO}: {e}",
            file=sys.stderr,
        )
        cache_root.mkdir(parents=True, exist_ok=True)

    return cache_root


def write_guiclaw_config(
    base_url: str, model: str, max_steps: int, api_key: str
) -> Path:
    """Write ~/.guiclaw/config.yaml for the GUIClaw CLI."""
    try:
        import yaml
    except ImportError:
        yaml = None  # type: ignore[assignment]

    config_dir = Path.home() / ".guiclaw"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "provider": {
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        },
        "max_steps": max_steps,
        "agent_profile": "default",
    }

    config_path = config_dir / "config.yaml"
    if yaml is not None:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        # Minimal YAML without the yaml package
        lines = [
            "provider:",
            f"  base_url: \"{base_url}\"",
            f"  model: \"{model}\"",
            f"  api_key: \"{api_key}\"",
            f"max_steps: {max_steps}",
            "agent_profile: default",
        ]
        config_path.write_text("\n".join(lines) + "\n")

    print(f"GUIClaw config written to {config_path}", file=sys.stderr)
    return config_path


def run_guiclaw_cli(task: str, backend: str, use_json: bool = True) -> dict:
    """Invoke the guiclaw CLI and capture results."""
    cmd = ["guiclaw", f"--backend={backend}"]
    if use_json:
        cmd.append("--json")
    cmd.append(task)

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return {
            "error": "guiclaw CLI not found. Ensure nanobot-ai is installed.",
            "exit_code": 127,
        }
    except subprocess.TimeoutExpired:
        return {
            "error": "guiclaw timed out after 600s",
            "exit_code": 124,
        }

    output = {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }

    # Try to parse JSON output
    if use_json and result.stdout.strip():
        try:
            output["parsed"] = json.loads(result.stdout)
        except json.JSONDecodeError:
            pass

    return output


def collect_artifacts(output_dir: str) -> dict:
    """Collect trajectory and memory artifacts from GUIClaw's data dirs."""
    artifacts = {}

    # Trajectory runs
    gui_runs = Path.home() / ".guiclaw" / "gui_runs"
    if gui_runs.exists():
        runs = sorted(
            [d for d in gui_runs.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if runs:
            latest = runs[0]
            dest = Path(output_dir) / "latest_run"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(latest, dest)
            artifacts["trajectory_dir"] = str(dest)
            artifacts["trajectory_files"] = [
                f.name for f in dest.rglob("*") if f.is_file()
            ]

    # Memory store
    memory_dir = Path.home() / ".guiclaw" / "memory"
    if memory_dir.exists() and any(memory_dir.iterdir()):
        dest = Path(output_dir) / "memory_snapshot"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(memory_dir, dest)
        artifacts["memory_dir"] = str(dest)
        artifacts["has_memory"] = True
    else:
        artifacts["has_memory"] = False

    # Skill library
    skills_file = Path.home() / ".guiclaw" / "skill" / "skills.py"
    if skills_file.exists():
        dest = Path(output_dir) / "skills.py"
        shutil.copy2(skills_file, dest)
        artifacts["skills_file"] = str(dest)
        artifacts["has_skills"] = True
    else:
        artifacts["has_skills"] = False

    return artifacts


def main():
    parser = argparse.ArgumentParser(
        description="KnowAct-GUIClaw: GUI agent with self-evolving memory and skills"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Describe the current screen layout and list all visible UI elements",
        help="Task description for the GUI agent",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("GUICLAW_BACKEND", "dry-run"),
        choices=["dry-run", "adb", "local", "ios", "hdc"],
        help="Device backend (default: dry-run)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GUICLAW_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        help="VLM model name for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("GUICLAW_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GUICLAW_API_KEY", "sk-local"),
        help="API key for the VLM endpoint",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("GUICLAW_MAX_STEPS", "15")),
        help="Maximum GUI agent steps per task",
    )
    args = parser.parse_args()

    # Set up output dir
    out_dir = resolve_output_path(
        "knowact_guiclaw_output/", default_filename="knowact_guiclaw_output/"
    ).rstrip("/")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Download sample data (for reference even if dry-run doesn't need it)
    sample_dir = ensure_sample_data()
    print(f"Sample data dir: {sample_dir}", file=sys.stderr)

    # Write GUIClaw config
    write_guiclaw_config(args.base_url, args.model, args.max_steps, args.api_key)

    # Run the task
    print(f"\n--- Running GUIClaw ({args.backend}) ---", file=sys.stderr)
    print(f"Task: {args.task}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Max steps: {args.max_steps}", file=sys.stderr)

    result = run_guiclaw_cli(args.task, args.backend)

    # Collect trajectory and memory artifacts
    artifacts = collect_artifacts(out_dir)

    # Build combined output
    output = {
        "task": args.task,
        "backend": args.backend,
        "model": args.model,
        "max_steps": args.max_steps,
        "result": result,
        "artifacts": artifacts,
    }

    # Save result
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save metrics
    metrics = {
        "task": args.task,
        "backend": args.backend,
        "model": args.model,
        "exit_code": result.get("exit_code", -1),
        "has_memory": artifacts.get("has_memory", False),
        "has_skills": artifacts.get("has_skills", False),
        "trajectory_files": len(artifacts.get("trajectory_files", [])),
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResult saved to {result_path}", file=sys.stderr)
    print(f"Metrics saved to {metrics_path}", file=sys.stderr)

    if result.get("exit_code", -1) != 0:
        err = result.get("error") or result.get("stderr", "")
        print(f"GUIClaw exited with code {result['exit_code']}: {err}", file=sys.stderr)
    else:
        print("GUIClaw completed successfully.", file=sys.stderr)

    # Print result summary to stdout
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
