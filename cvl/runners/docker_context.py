"""Docker Context runner for CVL examples.

Runs CVL examples on a remote machine with Docker installed.

This runner:
1. Syncs the example directory and CVL code to the remote machine via rsync
2. SSHs to the remote and runs the CVL script there
3. Docker commands run on the remote machine (where files exist)
4. Syncs outputs back to local machine

This is similar to SSHRunner but:
- Uses rsync for efficient file sync (delta transfer)
- Doesn't require CVL to be installed on remote (we sync it)
- Better suited for "bring your own GPU VM" workflows

The name "DockerContext" refers to the use case: you have a Docker-capable
machine accessible via SSH, similar to setting up a Docker context.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, List


class DockerContextRunner:
    """
    Run CVL examples on a remote Docker daemon.

    This runner:
    1. Syncs the example directory to the remote machine
    2. Sets up Docker to use the remote daemon (via SSH)
    3. Runs the CVL script (docker commands go to remote)
    4. Syncs outputs back to local machine

    Compared to SSHRunner:
    - SSHRunner: SSH to remote, run script there
    - DockerContextRunner: Run script locally, docker commands go to remote

    The main advantage is that you can use your local CVL installation and scripts,
    but execution happens on the remote GPU machine.

    Requirements:
    - SSH access to remote machine (key-based auth recommended)
    - Docker installed on remote machine
    - rsync installed locally and on remote

    Example:
        runner = DockerContextRunner()
        runner.run(
            example="nanogpt",
            preset="train",
            args=["--max_iters=1000"],
            host="ubuntu@gpu-server",
        )
    """

    # Default remote working directory
    DEFAULT_REMOTE_WORKDIR = "/tmp/cvl-remote"

    def __init__(self, ssh_key_path: Optional[str] = None):
        """
        Initialize Docker Context runner.

        Args:
            ssh_key_path: Path to SSH private key (optional, uses default if not specified)
        """
        self.ssh_key_path = ssh_key_path

    def run(
        self,
        example: str,
        preset: str,
        args: List[str],
        host: str,
        remote_workdir: Optional[str] = None,
        sync_outputs: bool = True,
        local_output_dir: Optional[str] = None,
        timeout_minutes: Optional[int] = None,
    ) -> int:
        """
        Run CVL example on remote Docker daemon.

        Args:
            example: Example name (e.g., "nanogpt")
            preset: Preset name (e.g., "train")
            args: Additional arguments for the training script
            host: Remote host in SSH format (e.g., "ubuntu@192.168.1.100" or SSH config alias)

            remote_workdir: Working directory on remote machine (default: /tmp/cvl-remote)
            sync_outputs: Whether to sync outputs back after completion (default: True)
            local_output_dir: Local directory for outputs (default: example/outputs/)
            timeout_minutes: Optional timeout in minutes

        Returns:
            Exit code (0 for success)
        """
        remote_workdir = remote_workdir or self.DEFAULT_REMOTE_WORKDIR

        try:
            # Find example directory locally
            example_dir = self._find_example(example)
            cvl_root = Path(__file__).parent.parent.parent
            print(f"Found example: {example_dir}")

            # Step 1: Sync files to remote
            print(f"Syncing files to {host}:{remote_workdir}...")
            self._sync_to_remote(host, example_dir, cvl_root, remote_workdir)

            # Step 2: Run the CVL script on remote via SSH
            exit_code = self._run_script(
                host=host,
                example_dir=example_dir,
                example=example,
                preset=preset,
                args=args,
                remote_workdir=remote_workdir,
                timeout_minutes=timeout_minutes,
            )

            # Step 3: Sync outputs back
            if sync_outputs and exit_code == 0:
                output_dir = local_output_dir or str(example_dir / "outputs")
                print(f"Syncing outputs to {output_dir}...")
                self._sync_from_remote(host, remote_workdir, example, output_dir)

            return exit_code

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 130

        except Exception as e:
            print(f"Error: {e}")
            return 1

    def _find_example(self, example: str) -> Path:
        """Find example directory by name."""
        cvl_root = Path(__file__).parent.parent.parent
        examples_dir = cvl_root / "examples"

        if not examples_dir.exists():
            raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

        # Search recursively for example.yaml with matching name
        import yaml
        for yaml_path in examples_dir.rglob("example.yaml"):
            try:
                with open(yaml_path) as f:
                    config = yaml.safe_load(f)
                if config.get("name") == example:
                    return yaml_path.parent
            except Exception:
                continue

        raise FileNotFoundError(f"Example not found: {example}")

    def _sync_to_remote(
        self,
        host: str,
        example_dir: Path,
        cvl_root: Path,
        remote_workdir: str,
    ):
        """Sync example and CVL code to remote machine."""
        ssh_opts = self._get_ssh_opts()

        # Create remote directory
        self._ssh_cmd(host, f"mkdir -p {remote_workdir}/example {remote_workdir}/cvlization")

        # Sync example directory
        example_rsync = [
            "rsync", "-az", "--delete",
            "-e", f"ssh {ssh_opts}",
            f"{example_dir}/",
            f"{host}:{remote_workdir}/example/",
        ]
        result = subprocess.run(example_rsync, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to sync example: {result.stderr}")

        # Sync cvlization package (for PYTHONPATH)
        cvl_package = cvl_root / "cvlization"
        if cvl_package.exists():
            cvl_rsync = [
                "rsync", "-az", "--delete",
                "-e", f"ssh {ssh_opts}",
                f"{cvl_package}/",
                f"{host}:{remote_workdir}/cvlization/",
            ]
            result = subprocess.run(cvl_rsync, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: Failed to sync cvlization package: {result.stderr}")

        print("File sync complete")

    def _sync_from_remote(
        self,
        host: str,
        remote_workdir: str,
        example: str,
        local_output_dir: str,
    ):
        """Sync outputs from remote machine."""
        ssh_opts = self._get_ssh_opts()
        os.makedirs(local_output_dir, exist_ok=True)

        # Common output directories
        for subdir in ["outputs", "logs", "checkpoints"]:
            remote_path = f"{host}:{remote_workdir}/example/{subdir}/"
            local_path = f"{local_output_dir}/{subdir}/"

            # Check if remote dir exists
            check = self._ssh_cmd(host, f"test -d {remote_workdir}/example/{subdir}", check=False)
            if check.returncode != 0:
                continue

            os.makedirs(local_path, exist_ok=True)
            rsync_cmd = [
                "rsync", "-az",
                "-e", f"ssh {ssh_opts}",
                remote_path,
                local_path,
            ]
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Synced {subdir}/")

    def _run_script(
        self,
        host: str,
        example_dir: Path,
        example: str,
        preset: str,
        args: List[str],
        remote_workdir: str,
        timeout_minutes: Optional[int],
    ) -> int:
        """Run the CVL script on the remote machine via SSH."""
        import shlex
        import yaml

        # Read example.yaml to find script
        example_yaml = example_dir / "example.yaml"
        with open(example_yaml) as f:
            config = yaml.safe_load(f)

        presets = config.get("presets", {})
        preset_info = presets.get(preset, {})

        if isinstance(preset_info, str):
            script_name = preset_info
        elif isinstance(preset_info, dict):
            script_name = preset_info.get("script", f"{preset}.sh")
        else:
            script_name = f"{preset}.sh"

        # Build remote command
        remote_example_dir = f"{remote_workdir}/example"
        safe_args = [shlex.quote(arg) for arg in args]
        args_str = " ".join(safe_args)

        # The remote command: cd to example dir, run script
        remote_cmd = f"cd {remote_example_dir} && bash {script_name} {args_str}"

        if timeout_minutes:
            remote_cmd = f"timeout {timeout_minutes}m bash -c '{remote_cmd}'"

        print(f"Running on {host}: {script_name} {args_str}")
        print("-" * 50)

        # SSH to remote and run
        ssh_opts = self._get_ssh_opts().split()
        ssh_cmd = ["ssh"] + ssh_opts + ["-t", host, remote_cmd]

        start_time = time.time()
        result = subprocess.run(ssh_cmd)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"\nCompleted in {int(duration)}s")
        elif result.returncode == 124:
            print(f"\nTimeout after {timeout_minutes} minutes")
        else:
            print(f"\nFailed with exit code {result.returncode}")

        return result.returncode

    def _ssh_cmd(self, host: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run SSH command on remote host."""
        ssh_opts = self._get_ssh_opts()
        full_cmd = ["ssh"] + ssh_opts.split() + [host, cmd]
        return subprocess.run(full_cmd, capture_output=True, text=True, check=False)

    def _get_ssh_opts(self) -> str:
        """Get SSH options string."""
        opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.ssh_key_path:
            opts.extend(["-i", self.ssh_key_path])
        return " ".join(opts)
