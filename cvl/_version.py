"""Version information for cvl package."""

__version__ = "0.1.0"
__git_hash__ = "04ba57f"
__repo_url__ = "https://github.com/kungfuai/CVlization"

def get_version_info():
    """Get full version information including git hash."""
    return f"{__version__}+git.{__git_hash__}"

def get_full_version():
    """Get version string with all information."""
    return f"cvl {__version__} (commit {__git_hash__})"
