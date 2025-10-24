"""Run command - execute example presets."""
import os
import sys
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def get_preset_info(example: Dict, preset_name: str) -> Optional[Dict]:
    """Get information about a specific preset.

    Args:
        example: Example metadata dict
        preset_name: Name of the preset to look up

    Returns:
        Dict with preset info (script, description), or None if not found
    """
    presets = example.get("presets", [])

    # Handle new dict format
    if isinstance(presets, dict):
        preset_data = presets.get(preset_name)
        if preset_data is None:
            return None

        # Support both dict and simple string values
        if isinstance(preset_data, dict):
            result = {
                "script": preset_data.get("script", f"{preset_name}.sh"),
                "description": preset_data.get("description", ""),
            }
            # Include optional 'command' field for CVL docker mode
            if "command" in preset_data:
                result["command"] = preset_data["command"]
            return result
        else:
            # Simple string value is treated as script name
            return {"script": str(preset_data), "description": ""}

    # Handle old list format - use convention
    elif isinstance(presets, list):
        if preset_name in presets:
            return {"script": f"{preset_name}.sh", "description": ""}

    return None


def find_script(example_path: str, script_name: str) -> Optional[str]:
    """Find the script file in the example directory.

    Args:
        example_path: Absolute path to example directory
        script_name: Script filename (e.g., "train.sh")

    Returns:
        Absolute path to script if found, None otherwise
    """
    script_path = os.path.join(example_path, script_name)
    if os.path.isfile(script_path):
        return script_path
    return None


def check_docker_running() -> Tuple[bool, str]:
    """Check if Docker daemon is running.

    Returns:
        Tuple of (is_running, error_message)
        is_running is True if Docker is accessible, error_message is empty if running
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return (True, "")
        else:
            return (False, "Docker daemon is not responding")
    except FileNotFoundError:
        return (False, "Docker command not found. Is Docker installed?")
    except subprocess.TimeoutExpired:
        return (False, "Docker command timed out. Is Docker running?")
    except Exception as e:
        return (False, f"Failed to check Docker status: {e}")


def check_docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally.

    Args:
        image_name: Name of the Docker image

    Returns:
        True if image exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_docker_image_name(example_path: str) -> Optional[str]:
    """Extract Docker image name from example directory.

    Uses the directory name as the image name (following build.sh convention).

    Args:
        example_path: Absolute path to example directory

    Returns:
        Image name, or None if cannot determine
    """
    return Path(example_path).name


def find_matching_examples(examples: List[Dict], identifier: str) -> List[Dict]:
    """Find examples matching identifier (exact or suffix match).

    Supports flexible matching:
    - Exact: "perception/vision_language/moondream2"
    - Short: "moondream2" matches "perception/vision_language/moondream2"
    - Partial: "line_detection/torch" matches "perception/line_detection/torch"

    Args:
        examples: List of example metadata dicts
        identifier: Example identifier (full path, partial path, or short name)

    Returns:
        List of matching examples (empty if none found, multiple if ambiguous)
    """
    normalized = identifier.removeprefix("examples/").rstrip("/")
    matches = []

    for example in examples:
        path = example.get("_path", "").removeprefix("examples/").rstrip("/")

        # Exact match - return immediately
        if path == normalized:
            return [example]

        # Suffix match - path ends with identifier
        if path.endswith("/" + normalized):
            matches.append(example)

    return matches


def get_example_path(examples: List[Dict], example_identifier: str) -> Optional[str]:
    """Get the absolute path to an example directory.

    Args:
        examples: List of example metadata dicts
        example_identifier: Example path (full, partial, or short name)

    Returns:
        Absolute path to example directory, or None if not found
    """
    from cvl.core.discovery import find_repo_root

    matches = find_matching_examples(examples, example_identifier)

    if len(matches) == 1:
        # Single match - return its path
        repo_root = find_repo_root()
        rel_path = matches[0].get("_path")
        return str(repo_root / rel_path)

    return None


def run_script(
    script_path: str,
    extra_args: List[str],
    no_live: bool = False,
    job_name: str = "",
    image_name: str = "",
    work_dir: Optional[str] = None
) -> Tuple[int, str]:
    """Execute a script with optional arguments.

    Args:
        script_path: Absolute path to script
        extra_args: Additional arguments to pass to script
        no_live: Disable live status display
        job_name: Name of the job (for live display)
        image_name: Docker image name (for live display)
        work_dir: Working directory for inputs/outputs (defaults to cwd if None)

    Returns:
        Tuple of (exit_code, error_message)
        Exit code 0 means success, error_message is empty on success
    """
    if not os.path.isfile(script_path):
        return (1, f"Script not found: {script_path}")

    if not os.access(script_path, os.X_OK):
        return (1, f"Script not executable: {script_path}")

    # Get the example directory to run script from
    example_dir = os.path.dirname(script_path)

    # Check if rich is available and not disabled
    use_live = not no_live
    if use_live:
        try:
            from rich.live import Live
            from rich.panel import Panel
            from rich.console import Console
        except ImportError:
            use_live = False  # Fallback to simple mode if rich not available

    # Track execution time
    start_time = time.time()

    try:
        # Run script from its directory using basename since cwd is set
        script_name = os.path.basename(script_path)

        # Generate deterministic container name for easy debugging
        # Format: {example}-{timestamp}
        # Use only last 6 digits of timestamp to keep name short
        timestamp_short = str(int(start_time))[-6:]
        # Extract just the example name from job_name (e.g., "moondream2" from "moondream2 predict")
        example_short = job_name.split()[0] if job_name else "job"
        container_name = f"{example_short}-{timestamp_short}"

        # Use env to set PYTHONUNBUFFERED for all Python scripts
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CVL_CONTAINER_NAME"] = container_name

        # Set CVL_WORK_DIR (default to cwd if not provided)
        if work_dir:
            env["CVL_WORK_DIR"] = str(Path(work_dir).resolve())
        else:
            env["CVL_WORK_DIR"] = os.getcwd()

        if use_live:
            # Live mode with rich
            from rich.live import Live
            from rich.panel import Panel
            from rich.console import Console

            console = Console(highlight=False, markup=False)

            # Run with Popen to stream output
            process = subprocess.Popen(
                ["bash", script_name] + extra_args,
                cwd=example_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )

            # Create status panel with container name
            status_text = f"▸ {job_name} ({image_name})\n📦 {container_name}\nStarting..."
            with Live(
                Panel(status_text, title="CVL Status", border_style="cyan"),
                refresh_per_second=2,
                console=console
            ) as live:
                # Stream output and update status
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    console.print(line, end='')

                    # Update status with elapsed time
                    elapsed = time.time() - start_time
                    duration_str = _format_duration(elapsed)
                    status_text = f"▸ {job_name} ({image_name})\n📦 {container_name}\n⏱  {duration_str} elapsed"
                    live.update(Panel(status_text, title="CVL Status", border_style="cyan"))

                process.wait()
                return_code = process.returncode

            # Calculate final duration
            duration = time.time() - start_time
            duration_str = _format_duration(duration)

            # Show completion message
            if return_code == 0:
                console.print(f"\n✓ {duration_str}", style="green bold")
            else:
                console.print(f"\n✗ Failed after {duration_str}", style="red bold")

            return (return_code, "")

        else:
            # Simple mode (no rich)
            result = subprocess.run(
                ["bash", script_name] + extra_args,
                cwd=example_dir,
                check=False,
                env=env,
            )

            # Calculate duration
            duration = time.time() - start_time
            duration_str = _format_duration(duration)

            # Show completion message
            if result.returncode == 0:
                print(f"\n✓ {duration_str}")
            else:
                print(f"\n✗ Failed after {duration_str}")

            return (result.returncode, "")

    except KeyboardInterrupt:
        duration = time.time() - start_time
        duration_str = _format_duration(duration)
        print(f"\n✗ Cancelled by user after {duration_str}")
        return (130, "")  # Standard exit code for SIGINT
    except Exception as e:
        return (1, f"Failed to execute script: {e}")


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "5m 32s" or "45s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def _prompt_user_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no response.

    Args:
        question: Question to ask
        default: Default answer if user just presses Enter

    Returns:
        True for yes, False for no
    """
    prompt_suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{question} {prompt_suffix} ").strip().lower()

    if not response:
        return default

    return response in ["y", "yes"]


def _run_via_cvl_docker_DEPRECATED(
    example: Dict,
    preset_info: Dict,
    inputs: Optional[str],
    outputs: Optional[str],
    extra_args: List[str],
) -> int:
    """DEPRECATED: Run preset via CVL-managed docker with explicit mounts.

    ⚠️ DEPRECATED: This function is deprecated and should not be used.

    Use the work_dir pattern instead where predict.sh scripts handle docker
    themselves and CVL just sets CVL_WORK_DIR environment variable.

    This old "Cog-like" execution path had CVL own the docker run command
    and mount user-specified input/output directories. It also used
    data/container_cache which is no longer needed.

    Deprecated in favor of:
    - predict.sh scripts that handle their own docker run
    - Single CVL_WORK_DIR instead of separate inputs/outputs
    - ${HOME}/.cache/huggingface instead of data/container_cache

    Args:
        example: Example metadata dict
        preset_info: Preset metadata dict with 'command' field
        inputs: Path to inputs directory (absolute or relative to cwd)
        outputs: Path to outputs directory (absolute or relative to cwd)
        extra_args: Additional arguments to pass to the command

    Returns:
        Exit code from docker run
    """
    from cvl.core.discovery import find_repo_root

    # Get example directory
    repo_root = find_repo_root()
    example_rel_path = example.get('_path', '')
    example_dir = repo_root / example_rel_path

    # Get image name
    image_name = example.get('image', Path(example_rel_path).name)

    # Get command to run
    command = preset_info.get('command')
    if not command:
        return (1, "No 'command' field in preset - cannot run via CVL docker mode")

    # Resolve input path
    if inputs:
        inputs_abs = Path(inputs).resolve()
        if not inputs_abs.exists():
            print(f"✗ Input directory not found: {inputs_abs}")
            return 1
    else:
        inputs_abs = None

    # Resolve output path with smart defaults
    if outputs:
        outputs_abs = Path(outputs).resolve()
    else:
        # Default: ./cvl-outputs/<example-name>/<timestamp>
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outputs_abs = Path.cwd() / "cvl-outputs" / example['name'] / timestamp

    # Create outputs directory
    outputs_abs.mkdir(parents=True, exist_ok=True)

    # Build docker command with security defaults
    docker_cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--read-only",
        "--workdir", "/workspace",
        # Mount example directory as workspace
        "--mount", f"type=bind,src={example_dir},dst=/workspace,readonly",
        # Mount cvlization package for dual-mode helpers
        "--mount", f"type=bind,src={repo_root},dst=/cvlization_repo,readonly",
        # Mount outputs (read-write)
        "--mount", f"type=bind,src={outputs_abs},dst=/mnt/cvl/outputs",
        # Tmpfs for temporary files
        "--mount", "type=tmpfs,dst=/tmp",
        # Environment variables
        "--env", "PYTHONUNBUFFERED=1",
        "--env", "PYTHONPATH=/cvlization_repo",
        "--env", "CVL_OUTPUTS=/mnt/cvl/outputs",
        "--env", "HF_HOME=/cache/huggingface",
        "--env", "HF_HUB_CACHE=/cache/huggingface/hub",
        "--env", "HF_DATASETS_CACHE=/cache/huggingface/datasets",
        "--env", "TRANSFORMERS_CACHE=/cache/huggingface/hub",
        "--env", "TORCH_HOME=/cache/torch",
    ]

    # Add inputs mount if provided
    if inputs_abs:
        docker_cmd.extend([
            "--mount", f"type=bind,src={inputs_abs},dst=/mnt/cvl/inputs,ro",
            "--env", "CVL_INPUTS=/mnt/cvl/inputs",
        ])

    # Add GPU support if needed
    if example.get('resources', {}).get('gpu'):
        docker_cmd.extend(["--gpus", "all"])

    # Add cache mount (optional - use repo's container_cache)
    repo_cache = repo_root / "data" / "container_cache"
    if repo_cache.exists():
        docker_cmd.extend([
            "--mount", f"type=bind,src={repo_cache},dst=/cache",
        ])

    # Add image and command
    docker_cmd.append(image_name)
    docker_cmd.extend(["bash", "-c", f"{command} {' '.join(extra_args)}"])

    # Show what we're running
    print(f"Running {example['name']} via CVL docker mode...")
    print(f"Inputs:  {inputs_abs or '(none)'}")
    print(f"Outputs: {outputs_abs}")
    print()

    # Track execution time
    start_time = time.time()

    try:
        result = subprocess.run(docker_cmd)

        # Calculate duration
        duration = time.time() - start_time
        duration_str = _format_duration(duration)

        # Show completion message
        if result.returncode == 0:
            print(f"\n✓ Completed in {duration_str}")
            print(f"✓ Outputs saved to: {outputs_abs}")
        else:
            print(f"\n✗ Failed after {duration_str}")

        return result.returncode

    except KeyboardInterrupt:
        duration = time.time() - start_time
        duration_str = _format_duration(duration)
        print(f"\n✗ Cancelled by user after {duration_str}")
        return 130


def run_example(
    examples: List[Dict],
    example_identifier: str,
    preset_name: str,
    extra_args: Optional[List[str]] = None,
    work_dir: Optional[str] = None,
    no_live: bool = False,
) -> Tuple[int, str]:
    """Run an example with a specific preset.

    Args:
        examples: All examples from discovery
        example_identifier: Example path (e.g., "generative/minisora")
        preset_name: Preset to run (e.g., "train")
        extra_args: Additional arguments to pass to script
        work_dir: Working directory for inputs/outputs (defaults to cwd if None)
        no_live: Disable live status display (default: False, use rich live mode)

    Returns:
        Tuple of (exit_code, error_message)
        Exit code 0 means success, error_message is empty on success
    """
    if extra_args is None:
        extra_args = []

    # Check if Docker is running
    docker_running, docker_error = check_docker_running()
    if not docker_running:
        return (1, f"✗ Docker is not running\n{docker_error}")

    # Find matching examples
    matches = find_matching_examples(examples, example_identifier)

    if len(matches) == 0:
        return (1, f"✗ No example found for: {example_identifier}")
    elif len(matches) > 1:
        # Ambiguous - show all matches
        paths = "\n  • ".join([ex.get("_path", "").removeprefix("examples/") for ex in matches])
        return (1, f"✗ Multiple examples found for '{example_identifier}':\n  • {paths}\n\nUse a more specific path to disambiguate.")

    # Single match found
    example = matches[0]
    example_path = get_example_path(examples, example_identifier)

    # Get preset info
    preset_info = get_preset_info(example, preset_name)
    if preset_info is None:
        available = _get_available_presets(example)
        return (1, f"Preset '{preset_name}' not found. Available: {available}")

    # Note: CVL docker mode (_run_via_cvl_docker_DEPRECATED) is deprecated in favor of
    # the work_dir pattern where scripts handle docker themselves.
    # The deprecated function is kept for reference but is no longer called.
    #
    # Old pattern (DEPRECATED):
    #   - CVL owned docker run command
    #   - Used --inputs/--outputs flags
    #   - Mounted data/container_cache to /cache
    #
    # New pattern (CURRENT):
    #   - predict.sh handles docker run
    #   - Uses -w/--work-dir flag (defaults to cwd)
    #   - Mounts ${HOME}/.cache/huggingface

    # Standalone mode: run the script (which calls docker itself)
    # Find the script
    script_name = preset_info["script"]
    script_path = find_script(example_path, script_name)
    if script_path is None:
        return (1, f"Script not found: {script_name} in {example_path}")

    # Check if Docker image exists (except for build preset)
    if preset_name != "build":
        image_name = get_docker_image_name(example_path)
        if image_name and not check_docker_image_exists(image_name):
            # Check if build preset exists
            build_preset = get_preset_info(example, "build")

            print(f"✗ Docker image '{image_name}' not found\n")
            print(f"Build it first:")
            print(f"  cvl run {example_identifier} build")
            print(f"  or")
            print(f"  bash {example_path}/{script_name.replace(preset_name, 'build') if preset_name in script_name else 'build.sh'}\n")

            # Offer to build now if build preset exists
            if build_preset:
                if _prompt_user_yes_no("Build it now?"):
                    print()  # Blank line before build output
                    exit_code, error_msg = run_example(
                        examples,
                        example_identifier,
                        "build",
                        []
                    )
                    if exit_code != 0:
                        return (exit_code, "Build failed")
                    print()  # Blank line after build
                else:
                    return (1, "Cancelled by user")
            else:
                return (1, "Docker image not found")

    # Show what we're running
    example_name = example.get("name", Path(example_path).name)
    example_full_path = example.get("_path", "").removeprefix("examples/")
    image_name = get_docker_image_name(example_path)

    print(f"Running {example_name} {preset_name}...")

    # Show full path if user provided short/partial name
    if example_full_path != example_identifier.removeprefix("examples/").rstrip("/"):
        print(f"Example: {example_full_path} (matched: {example_identifier})")
    else:
        print(f"Example: {example_full_path}")

    print(f"Docker:  {image_name}")
    print(f"Script:  {script_name}")
    print()  # Blank line before script output

    # Run the script
    exit_code, error_msg = run_script(
        script_path,
        extra_args,
        no_live=no_live,
        job_name=f"{example_name} {preset_name}",
        image_name=image_name,
        work_dir=work_dir
    )

    # Show path mappings after completion (if successful)
    if exit_code == 0:
        print("\nPath Mappings (Container → Host):")
        print(f"  /workspace → {example_path}")

        # Show work directory if CVL_WORK_DIR was set
        if work_dir:
            work_dir_abs = Path(work_dir).resolve()
            print(f"  /mnt/cvl/workspace → {work_dir_abs}")
        else:
            # Show current directory as default work directory
            print(f"  /mnt/cvl/workspace → {os.getcwd()}")

        # Show outputs directory for standalone mode
        outputs_dir = Path(example_path) / "outputs"
        if outputs_dir.exists():
            print(f"  /workspace/outputs → {outputs_dir}")

    return (exit_code, error_msg)


def _get_available_presets(example: Dict) -> str:
    """Get list of available presets as a string.

    Args:
        example: Example metadata dict

    Returns:
        Comma-separated list of preset names
    """
    presets = example.get("presets", [])
    if isinstance(presets, dict):
        return ", ".join(presets.keys())
    elif isinstance(presets, list):
        return ", ".join(presets)
    return "none"
