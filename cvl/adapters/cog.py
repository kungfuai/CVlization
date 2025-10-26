"""
Cog adapter for CVlization centralized caching.

This adapter enables Cog-based examples to use CVlization's centralized
caching pattern by setting environment variables that the predict.py can
use to configure cache directories.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class CogCacheAdapter:
    """Handle Cog execution with optional centralized caching.

    This adapter checks if a Cog example has configured centralized caching
    in its example.yaml, and if so, runs Cog commands with appropriate
    environment variables set to point to the shared cache directory.

    Example configuration in example.yaml:
        cog:
          enabled: true
          cache:
            huggingface: true  # Mount data/container_cache/huggingface -> HF_HOME
            custom:
              WEIGHTS_CACHE_DIR: huggingface/hub  # Custom env var
    """

    def __init__(self, repo_root: str):
        """Initialize the adapter.

        Args:
            repo_root: Path to CVlization repository root
        """
        self.repo_root = Path(repo_root)
        self.cache_base = self.repo_root / "data" / "container_cache"

    def should_use_cache(self, example_dir: str) -> bool:
        """Check if example has centralized caching configured.

        Args:
            example_dir: Path to example directory

        Returns:
            True if example.yaml has cog.cache configuration
        """
        example_yaml = Path(example_dir) / "example.yaml"
        if not example_yaml.exists():
            return False

        try:
            with open(example_yaml) as f:
                config = yaml.safe_load(f)

            # Check if cog.cache is configured
            if not isinstance(config, dict):
                return False

            cog_config = config.get("cog", {})
            if not isinstance(cog_config, dict):
                return False

            cache_config = cog_config.get("cache", {})
            return bool(cache_config)

        except Exception:
            return False

    def get_cache_env_vars(self, example_dir: str) -> Dict[str, str]:
        """Get environment variables for cache paths.

        Args:
            example_dir: Path to example directory

        Returns:
            Dictionary of environment variable name -> path value
        """
        example_yaml = Path(example_dir) / "example.yaml"
        env_vars = {}

        try:
            with open(example_yaml) as f:
                config = yaml.safe_load(f)

            cache_config = config.get("cog", {}).get("cache", {})

            # Handle HuggingFace cache
            if cache_config.get("huggingface"):
                hf_cache = self.cache_base / "huggingface"
                if hf_cache.exists():
                    env_vars["HF_HOME"] = str(hf_cache)

            # Handle custom cache mappings
            custom = cache_config.get("custom", {})
            for env_var, subpath in custom.items():
                cache_path = self.cache_base / subpath
                if cache_path.exists():
                    env_vars[env_var] = str(cache_path)

        except Exception as e:
            # Silently fail - will fallback to original behavior
            pass

        return env_vars

    def run_with_cache(
        self,
        example_dir: str,
        cog_command: str,
        extra_args: List[str],
        no_live: bool = False,
        job_name: str = "",
    ) -> Tuple[int, str]:
        """Run Cog command with centralized cache environment variables.

        Args:
            example_dir: Path to example directory
            cog_command: Cog command (build, predict, etc.)
            extra_args: Additional arguments to pass to Cog
            no_live: If True, capture output instead of streaming
            job_name: Optional job name for logging

        Returns:
            Tuple of (return_code, output_string)
        """
        # Get cache environment variables
        cache_env = self.get_cache_env_vars(example_dir)

        # Build command
        cmd = ["cog", cog_command] + extra_args

        # Prepare environment (inherit current + add cache vars)
        env = os.environ.copy()
        env.update(cache_env)

        # Log cache usage
        if cache_env:
            print(f"Using centralized cache:")
            for var, path in cache_env.items():
                print(f"  {var}={path}")

        # Run command
        if no_live:
            # Capture output
            try:
                result = subprocess.run(
                    cmd,
                    cwd=example_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                output = result.stdout + result.stderr
                return result.returncode, output
            except Exception as e:
                return 1, f"Error running cog: {e}"
        else:
            # Stream output
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=example_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                output_lines = []
                if process.stdout:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        output_lines.append(line)

                return_code = process.wait()
                return return_code, "".join(output_lines)

            except Exception as e:
                return 1, f"Error running cog: {e}"
