"""SkyPilot runner for CVL examples.

Runs CVL examples on any cloud (AWS, GCP, Azure, Lambda Labs, etc.) via SkyPilot.

SkyPilot handles:
- Multi-cloud instance provisioning
- Spot instance management and recovery
- Automatic setup and file syncing
- Cost optimization

Requirements:
- skypilot: pip install "skypilot[aws]" or "skypilot[gcp]" etc.
- Cloud credentials configured
"""

import atexit
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

# Lazy import skypilot
_sky = None


def _get_sky():
    """Lazy load skypilot."""
    global _sky
    if _sky is None:
        try:
            import sky
            _sky = sky
        except ImportError:
            print("skypilot not installed. Install with: pip install 'skypilot[aws]'")
            print("See: https://skypilot.readthedocs.io/en/latest/getting-started/installation.html")
            sys.exit(1)
    return _sky


class SkyPilotRunner:
    """
    Run CVL examples on any cloud via SkyPilot.

    SkyPilot abstracts cloud providers and handles:
    - Instance provisioning across AWS, GCP, Azure, Lambda Labs
    - Spot instance management with automatic recovery
    - File syncing to remote instances
    - Cost optimization

    Example:
        runner = SkyPilotRunner()
        runner.run(
            command="python train.py --max_iters=1000",
            workdir="./examples/nanogpt",
            gpu="A100:1",
        )
    """

    def __init__(self):
        """Initialize SkyPilot runner."""
        self.sky = _get_sky()
        self.current_cluster = None
        self._setup_cleanup_handlers()

    def run(
        self,
        command: str,
        workdir: Optional[str] = None,
        gpu: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        use_spot: bool = False,
        cluster_name: Optional[str] = None,
        idle_minutes_to_autostop: int = 10,
        down: bool = True,
        env: Optional[Dict[str, str]] = None,
        file_mounts: Optional[Dict[str, str]] = None,
        setup: Optional[str] = None,
    ) -> int:
        """
        Run a command on a cloud instance via SkyPilot.

        Args:
            command: Command to run (e.g., "python train.py --max_iters=1000")
            workdir: Local directory to sync to remote (default: current dir)

            gpu: GPU spec (e.g., "A100:1", "V100:4", "T4:1")
            cloud: Cloud provider (e.g., "aws", "gcp", "azure", "lambda")
            region: Cloud region (e.g., "us-east-1")
            use_spot: Use spot/preemptible instances (default: False)

            cluster_name: Name for the cluster (default: auto-generated)
            idle_minutes_to_autostop: Auto-stop after idle (default: 10)
            down: Tear down cluster after completion (default: True)

            env: Environment variables dict
            file_mounts: Additional files to mount {remote_path: local_path}
            setup: Setup commands to run before main command

        Returns:
            Exit code (0 for success)
        """
        sky = self.sky

        try:
            # Build task
            task = sky.Task(
                run=command,
                workdir=workdir,
                setup=setup,
                envs=env,
            )

            # Set resources
            resources_kwargs = {}
            if gpu:
                resources_kwargs["accelerators"] = gpu
            if cloud:
                resources_kwargs["cloud"] = getattr(sky.clouds, cloud.upper(), None) or sky.clouds.Cloud.from_str(cloud)
            if region:
                resources_kwargs["region"] = region
            if use_spot:
                resources_kwargs["use_spot"] = True

            if resources_kwargs:
                task.set_resources(sky.Resources(**resources_kwargs))

            # File mounts
            if file_mounts:
                task.set_file_mounts(file_mounts)

            # Generate cluster name
            if not cluster_name:
                cluster_name = f"cvl-{int(time.time())}"
            self.current_cluster = cluster_name

            print(f"Launching cluster: {cluster_name}")
            if gpu:
                print(f"  GPU: {gpu}")
            if cloud:
                print(f"  Cloud: {cloud}")
            if use_spot:
                print(f"  Spot: enabled")
            print("-" * 50)

            # Launch
            sky.launch(
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=idle_minutes_to_autostop,
                down=down,
                stream_logs=True,
            )

            print("-" * 50)
            print("Task completed successfully")
            self.current_cluster = None
            return 0

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 130

        except Exception as e:
            print(f"Error: {e}")
            return 1

        finally:
            # Cleanup if down=True and cluster still exists
            if down and self.current_cluster:
                self._teardown(self.current_cluster)
                self.current_cluster = None

    def _teardown(self, cluster_name: str):
        """Tear down cluster."""
        try:
            print(f"Tearing down cluster: {cluster_name}")
            self.sky.down(cluster_name)
        except Exception as e:
            print(f"Warning: Failed to tear down cluster: {e}")
            print(f"Run manually: sky down {cluster_name}")

    def _setup_cleanup_handlers(self):
        """Setup handlers to clean up on exit."""
        def cleanup():
            if self.current_cluster:
                self._teardown(self.current_cluster)

        atexit.register(cleanup)

        def signal_handler(signum, frame):
            cleanup()
            sys.exit(130)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # Convenience methods for common patterns

    def run_example(
        self,
        example_dir: str,
        script: str = "train.py",
        args: Optional[List[str]] = None,
        gpu: str = "A100:1",
        **kwargs,
    ) -> int:
        """
        Run a CVL example directory.

        Args:
            example_dir: Path to example directory
            script: Script to run (default: train.py)
            args: Arguments for the script
            gpu: GPU spec (default: A100:1)
            **kwargs: Additional arguments passed to run()

        Returns:
            Exit code
        """
        args_str = " ".join(args or [])
        command = f"python {script} {args_str}".strip()

        return self.run(
            command=command,
            workdir=example_dir,
            gpu=gpu,
            **kwargs,
        )
