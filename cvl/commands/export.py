"""Export command - bundle an example with the cvlization package."""
from __future__ import annotations

import shutil
import subprocess
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


def _write_export_readme(
    export_root: Path, rel_example_path: Path, commit: Optional[str]
) -> None:
    """Drop a small helper README into the export."""
    readme = export_root / "EXPORT_README.md"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # When inside the example directory, going up len(parts) levels lands at export_root.
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


def export_example(
    examples: List[Dict],
    example_identifier: str,
    dest: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[int, str]:
    """Export an example directory plus the cvlization package.

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
        suggested = ", ".join(m.get('_path', '') for m in matches if m.get('_path'))
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

    try:
        _copytree(example_src, export_root / rel_path, overwrite=overwrite)
        _copytree(repo_root / "cvlization", export_root / "cvlization", overwrite=overwrite)
    except Exception as e:
        return 1, f"Failed to copy files: {e}"

    _write_export_readme(export_root, rel_path, _git_commit(repo_root))

    return 0, f"Exported to {export_root}"
