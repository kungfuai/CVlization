"""Run command - execute example presets."""
import os
import subprocess
from typing import List, Dict, Optional, Tuple


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
            return {
                "script": preset_data.get("script", f"{preset_name}.sh"),
                "description": preset_data.get("description", ""),
            }
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


def get_example_path(examples: List[Dict], example_identifier: str) -> Optional[str]:
    """Get the absolute path to an example directory.

    Args:
        examples: List of example metadata dicts
        example_identifier: Example path (e.g., "generative/minisora")

    Returns:
        Absolute path to example directory, or None if not found
    """
    # Normalize path - remove leading "examples/" if present
    normalized_path = example_identifier.removeprefix("examples/").rstrip("/")

    for example in examples:
        example_rel_path = example.get("_path", "").removeprefix("examples/").rstrip("/")
        if example_rel_path == normalized_path:
            return example.get("_path")

    return None


def run_script(script_path: str, extra_args: List[str]) -> Tuple[int, str]:
    """Execute a script with optional arguments.

    Args:
        script_path: Absolute path to script
        extra_args: Additional arguments to pass to script

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

    try:
        # Run script from its directory
        result = subprocess.run(
            ["bash", script_path] + extra_args,
            cwd=example_dir,
            check=False,
        )
        return (result.returncode, "")
    except Exception as e:
        return (1, f"Failed to execute script: {e}")


def run_example(
    examples: List[Dict],
    example_identifier: str,
    preset_name: str,
    extra_args: Optional[List[str]] = None,
) -> Tuple[int, str]:
    """Run an example with a specific preset.

    Args:
        examples: All examples from discovery
        example_identifier: Example path (e.g., "generative/minisora")
        preset_name: Preset to run (e.g., "train")
        extra_args: Additional arguments to pass to script

    Returns:
        Tuple of (exit_code, error_message)
        Exit code 0 means success, error_message is empty on success
    """
    if extra_args is None:
        extra_args = []

    # Find the example
    example_path = get_example_path(examples, example_identifier)
    if example_path is None:
        return (1, f"Example not found: {example_identifier}")

    # Find the example metadata
    normalized_path = example_identifier.removeprefix("examples/").rstrip("/")
    example = None
    for ex in examples:
        ex_rel_path = ex.get("_path", "").removeprefix("examples/").rstrip("/")
        if ex_rel_path == normalized_path:
            example = ex
            break

    if example is None:
        return (1, f"Example metadata not found: {example_identifier}")

    # Get preset info
    preset_info = get_preset_info(example, preset_name)
    if preset_info is None:
        available = _get_available_presets(example)
        return (1, f"Preset '{preset_name}' not found. Available: {available}")

    # Find the script
    script_name = preset_info["script"]
    script_path = find_script(example_path, script_name)
    if script_path is None:
        return (1, f"Script not found: {script_name} in {example_path}")

    # Run the script
    return run_script(script_path, extra_args)


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
