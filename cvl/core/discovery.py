"""Discovery and loading of examples from the repository."""
from pathlib import Path
from typing import List, Dict, Optional
import os
import subprocess
import yaml

from cvl.core.config import get_repo_root_from_config, save_repo_root


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find CVlization repository root.

    Precedence order:
    1. Git clone (via git rev-parse) - if running inside CVlization repo
    2. CVLIZATION_ROOT environment variable
    3. Saved config from editable install (pip install -e .)
    4. Managed checkout in platform data directory
    5. Fail with helpful message

    Args:
        start_path: Starting directory (defaults to current working directory)

    Returns:
        Path to repository root (resolved, absolute)

    Raises:
        RuntimeError: If repository root not found
    """
    # 1. Check if we're inside a Git clone of CVlization
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start_path or Path.cwd(),
            capture_output=True,
            check=True,
            text=True,
        )
        repo_root = Path(result.stdout.strip()).resolve()
        if (repo_root / "examples").exists():
            # Save to config for future use (allows cvl to work from anywhere)
            # Only save if not already configured or if different location
            saved_root = get_repo_root_from_config()
            if saved_root != str(repo_root):
                try:
                    save_repo_root(str(repo_root))
                except (IOError, OSError):
                    # Silently fail if we can't write config (e.g., permissions)
                    pass
            return repo_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. Check CVLIZATION_ROOT environment variable
    if "CVLIZATION_ROOT" in os.environ:
        env_path = Path(os.environ["CVLIZATION_ROOT"]).resolve()
        if env_path.exists() and (env_path / "examples").exists():
            return env_path
        # Warn if set but invalid
        if env_path.exists():
            raise RuntimeError(
                f"CVLIZATION_ROOT is set to '{env_path}' but no examples/ directory found.\n"
                "Unset CVLIZATION_ROOT or point it to a valid CVlization repository."
            )

    # 3. Check saved config from editable install
    config_root = get_repo_root_from_config()
    if config_root:
        config_path = Path(config_root).resolve()
        if config_path.exists() and (config_path / "examples").exists():
            return config_path

    # 4. Check managed checkout location (platform-specific)
    # Note: Using simple approach; could use platformdirs library for production
    if os.name == 'nt':  # Windows
        data_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    elif os.path.exists(Path.home() / 'Library'):  # macOS
        data_dir = Path.home() / 'Library' / 'Application Support'
    else:  # Linux/Unix (XDG)
        data_dir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))

    managed_path = (data_dir / "CVlization" / "repo").resolve()
    if managed_path.exists() and (managed_path / "examples").exists():
        return managed_path

    # 5. Nothing found - fail with helpful message
    raise RuntimeError(
        "CVlization repository not found.\n\n"
        "Options:\n"
        "  1. Clone CVlization and install:\n"
        "     git clone https://github.com/kungfuai/CVlization\n"
        "     cd CVlization\n"
        "     pip install -e .\n"
        "  2. Set CVLIZATION_ROOT=/path/to/CVlization\n"
        f"  3. Clone to managed location: {managed_path}\n"
        "     (future: run 'cvl init' to do this automatically)"
    )


def load_example_yaml(example_dir: Path) -> Optional[Dict]:
    """Load and parse example.yaml from a directory.

    Args:
        example_dir: Path to example directory

    Returns:
        Parsed YAML as dict, or None if file doesn't exist or is invalid
    """
    yaml_path = example_dir / "example.yaml"
    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            # Add path for reference
            rel_path = str(example_dir.relative_to(find_repo_root()))
            data['_path'] = rel_path
            # Add type based on path (example or benchmark)
            if rel_path.startswith("benchmarks/"):
                data['_type'] = 'benchmark'
            else:
                data['_type'] = 'example'
            return data
    except (yaml.YAMLError, IOError):
        return None


def find_all_examples(repo_root: Optional[Path] = None) -> List[Dict]:
    """Find all examples with example.yaml files.

    Searches both examples/ and benchmarks/ directories.

    Args:
        repo_root: Repository root (auto-detected if not provided)

    Returns:
        List of example metadata dicts
    """
    if repo_root is None:
        repo_root = find_repo_root()

    examples = []

    # Search in examples/ directory
    examples_dir = repo_root / "examples"
    if examples_dir.exists():
        for yaml_file in examples_dir.rglob("example.yaml"):
            example = load_example_yaml(yaml_file.parent)
            if example:
                examples.append(example)

    # Search in benchmarks/ directory
    benchmarks_dir = repo_root / "benchmarks"
    if benchmarks_dir.exists():
        for yaml_file in benchmarks_dir.rglob("example.yaml"):
            example = load_example_yaml(yaml_file.parent)
            if example:
                examples.append(example)

    return examples
