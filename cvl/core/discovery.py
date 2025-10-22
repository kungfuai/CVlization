"""Discovery and loading of examples from the repository."""
from pathlib import Path
from typing import List, Dict, Optional
import os
import subprocess
import yaml


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find CVlization repository root.

    Precedence order:
    1. Git clone (via git rev-parse) - if running inside CVlization repo
    2. CVLIZATION_ROOT environment variable
    3. Managed checkout in platform data directory
    4. Fail with helpful message

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

    # 3. Check managed checkout location (platform-specific)
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

    # 4. Nothing found - fail with helpful message
    raise RuntimeError(
        "CVlization repository not found.\n\n"
        "Options:\n"
        "  1. Run from inside a CVlization git clone\n"
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
            data['_path'] = str(example_dir.relative_to(find_repo_root()))
            return data
    except (yaml.YAMLError, IOError):
        return None


def find_all_examples(repo_root: Optional[Path] = None) -> List[Dict]:
    """Find all examples with example.yaml files.

    Args:
        repo_root: Repository root (auto-detected if not provided)

    Returns:
        List of example metadata dicts
    """
    if repo_root is None:
        repo_root = find_repo_root()

    examples_dir = repo_root / "examples"
    if not examples_dir.exists():
        return []

    examples = []
    for yaml_file in examples_dir.rglob("example.yaml"):
        example = load_example_yaml(yaml_file.parent)
        if example:
            examples.append(example)

    return examples
