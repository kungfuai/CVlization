"""Version information for cvl package."""
import subprocess
from pathlib import Path

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError


def get_git_hash():
    """Get short git commit hash."""
    try:
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unknown"


def get_package_version():
    """Get version from package metadata (set by setuptools_scm)."""
    try:
        return version("cvl")
    except PackageNotFoundError:
        return "0.0.0+dev"


__version__ = get_package_version()
__git_hash__ = get_git_hash()
__repo_url__ = "https://github.com/kungfuai/CVlization"


def get_version_info():
    """Get full version information including git hash."""
    return f"{__version__}+git.{__git_hash__}"


def get_full_version():
    """Get version string with all information."""
    return f"cvl {__version__} (commit {__git_hash__})"
