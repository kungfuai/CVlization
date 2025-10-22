"""Discovery and loading of examples from the repository."""
from pathlib import Path
from typing import List, Dict, Optional
import yaml


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find repository root by looking for .git directory.

    Args:
        start_path: Starting directory (defaults to current working directory)

    Returns:
        Path to repository root

    Raises:
        RuntimeError: If repository root not found
    """
    current = start_path or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Not in a git repository. Run from CVlization directory.")


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
