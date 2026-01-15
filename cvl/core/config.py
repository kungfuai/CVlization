"""Configuration management for cvl CLI.

Supports hierarchical config for runners:
  CLI flags > Example config (cvl.yaml) > User config (~/.cvl/config.yaml) > Defaults
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Default runner configurations
RUNNER_DEFAULTS = {
    "sagemaker": {
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "volume_size_gb": 50,
        "max_run_minutes": 60,
        "spot": False,
        "download_outputs": True,
    },
    "k8s": {
        "gpu": 1,
        "cpu": "2",
        "memory": "8Gi",
        "namespace": "default",
    },
    "skypilot": {
        "gpu": "A10G:1",
        "use_spot": False,
    },
    "ssh": {
        "gpu_ids": None,
    },
}


def get_config_dir() -> Path:
    """Get the configuration directory for cvl.

    Uses XDG_CONFIG_HOME on Linux/macOS, LOCALAPPDATA on Windows.

    Returns:
        Path to config directory
    """
    if os.name == 'nt':  # Windows
        config_home = os.getenv('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))
    else:  # Linux, macOS
        config_home = os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))

    return Path(config_home) / 'cvl'


def get_config_file() -> Path:
    """Get the path to the config file.

    Returns:
        Path to config.json
    """
    return get_config_dir() / 'config.json'


def load_config() -> dict:
    """Load configuration from file.

    Returns:
        Configuration dict, or empty dict if file doesn't exist
    """
    config_file = get_config_file()

    if not config_file.exists():
        return {}

    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: dict) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dict to save
    """
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = get_config_file()

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def get_repo_root_from_config() -> Optional[str]:
    """Get the CVlization repo root from saved config.

    Returns:
        Absolute path to repo root, or None if not configured
    """
    config = load_config()
    repo_root = config.get('repo_root')

    if repo_root and Path(repo_root).exists():
        return repo_root

    return None


def save_repo_root(repo_root: str) -> None:
    """Save the CVlization repo root to config.

    Args:
        repo_root: Absolute path to CVlization repo root
    """
    config = load_config()
    config['repo_root'] = str(Path(repo_root).absolute())
    save_config(config)


# --- Runner configuration functions ---

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file, return empty dict if not found or yaml unavailable."""
    if not YAML_AVAILABLE:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception:
        return {}


def get_user_runner_config() -> Dict[str, Any]:
    """Load user runner config from ~/.cvl/config.yaml."""
    return load_yaml_file(Path.home() / ".cvl" / "config.yaml")


def get_example_runner_config(example_path: Path) -> Dict[str, Any]:
    """Load example config from cvl.yaml in example directory."""
    return load_yaml_file(example_path / "cvl.yaml")


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dicts, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_runner_config(
    runner: str,
    example_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get merged config for a runner.

    Priority: CLI overrides > Example config > User config > Defaults

    Args:
        runner: Runner name (sagemaker, k8s, skypilot, ssh)
        example_path: Path to example directory (for cvl.yaml)
        cli_overrides: Config values from CLI flags

    Returns:
        Merged config dict for the runner
    """
    # Start with defaults
    config = RUNNER_DEFAULTS.get(runner, {}).copy()

    # Merge user config
    user_config = get_user_runner_config()
    if runner in user_config:
        config = deep_merge(config, user_config[runner])

    # Merge example config
    if example_path:
        example_config = get_example_runner_config(example_path)
        if runner in example_config:
            config = deep_merge(config, example_config[runner])

    # Merge CLI overrides (filter out None values)
    if cli_overrides:
        filtered = {k: v for k, v in cli_overrides.items() if v is not None}
        config = deep_merge(config, filtered)

    return config


def validate_sagemaker_config(config: Dict[str, Any]) -> Optional[str]:
    """Validate SageMaker config, return error message if invalid."""
    if not config.get("output_path"):
        return "SageMaker requires --output-path (S3 path for outputs)"

    output_path = config["output_path"]
    if not output_path.startswith("s3://"):
        return f"output_path must be an S3 path (s3://...), got: {output_path}"

    # Check for role ARN
    role_arn = config.get("role_arn") or os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role_arn:
        return "SageMaker requires SAGEMAKER_ROLE_ARN environment variable or --role-arn flag"

    return None
