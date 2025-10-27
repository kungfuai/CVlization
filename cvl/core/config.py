"""Configuration management for cvl CLI."""
import json
import os
from pathlib import Path
from typing import Optional


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
