"""SSH-based remote execution for CVL examples."""

import shlex
import sys
from typing import Optional, List
from pathlib import Path


class SSHRunner:
    """Execute CVL examples on remote hosts via SSH."""

    def __init__(self, ssh_key_path: Optional[str] = None):
        """
        Initialize SSH runner.

        Args:
            ssh_key_path: Path to SSH private key (optional)
        """
        self.ssh_key_path = ssh_key_path
        self._ssh = None

    def run_remote(
        self,
        host: str,
        example: str,
        preset: str,
        args: List[str],
        timeout_minutes: Optional[int] = None,
        timeout_action: str = "sudo shutdown -h now",
        setup_cvl: bool = True
    ) -> int:
        """
        Run CVL example on remote host.

        Args:
            host: Remote host (e.g., "ubuntu@123.45.67.89" or "123.45.67.89")
            example: Example name (e.g., "nanogpt")
            preset: Preset name (e.g., "train")
            args: Additional arguments to pass to the script
            timeout_minutes: Optional timeout in minutes (kills job after timeout)
            timeout_action: Command to run if timeout occurs
                          WARNING: This is inserted into shell without escaping.
                          Only use trusted values. Intended for "sudo shutdown -h now".
            setup_cvl: Whether to setup CVL on remote (git clone, pip install)

        Returns:
            Exit code from remote command
        """
        # Validate timeout
        if timeout_minutes is not None and timeout_minutes <= 0:
            raise ValueError(f"timeout_minutes must be positive, got {timeout_minutes}")
        try:
            import paramiko
        except ImportError:
            print("❌ paramiko not installed. Install with: pip install paramiko")
            sys.exit(1)

        # Parse host
        if "@" in host:
            user, hostname = host.split("@", 1)
        else:
            user = "ubuntu"
            hostname = host

        # Connect via SSH
        print(f"🔌 Connecting to {user}@{hostname}...")
        ssh = paramiko.SSHClient()

        # WARNING: AutoAddPolicy accepts any host key (vulnerable to MITM)
        # For production use, consider using RejectPolicy and manually managing known_hosts
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(
                hostname,
                username=user,
                key_filename=self.ssh_key_path,
                timeout=30
            )
            print(f"✅ Connected to {hostname}")

            # Setup CVL if requested
            if setup_cvl:
                self._setup_cvl(ssh)

            # Build and execute command
            cmd = self._build_command(example, preset, args, timeout_minutes, timeout_action)
            print(f"🏃 Running: cvl run {example} {preset}")

            # Execute command
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)

            # Stream output in real-time
            for line in stdout:
                print(line, end='')

            # Get exit code
            exit_code = stdout.channel.recv_exit_status()

            if exit_code == 0:
                print(f"\n✅ Command completed successfully")
            elif exit_code == 124:
                print(f"\n⏱️ Command timed out after {timeout_minutes} minutes")
            else:
                print(f"\n⚠️ Command exited with code {exit_code}")

            return exit_code

        except paramiko.AuthenticationException:
            print(f"❌ Authentication failed for {user}@{hostname}")
            print(f"   Check SSH key or add your public key to remote ~/.ssh/authorized_keys")
            raise
        except paramiko.SSHException as e:
            print(f"❌ SSH error: {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
        finally:
            ssh.close()

    def _setup_cvl(self, ssh):
        """Setup CVL on remote machine if not already installed."""
        print("🔧 Checking CVL installation...")

        # Check if CVL is already installed
        stdin, stdout, stderr = ssh.exec_command("which cvl")
        if stdout.channel.recv_exit_status() == 0:
            print("✅ CVL already installed")
            return

        print("📦 Installing CVL...")

        # Clone CVlization repo
        setup_cmd = """
            if [ ! -d "$HOME/CVlization" ]; then
                echo "Cloning CVlization repository..."
                git clone https://github.com/kungfuai/CVlization "$HOME/CVlization"
            else
                echo "CVlization directory exists, updating..."
                cd "$HOME/CVlization" && git pull
            fi

            echo "Installing CVL..."
            cd "$HOME/CVlization"
            pip install -e . --quiet

            echo "CVL setup complete"
        """

        stdin, stdout, stderr = ssh.exec_command(setup_cmd)

        # Show output
        for line in stdout:
            print(f"  {line.rstrip()}")

        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            print("❌ Failed to setup CVL")
            err_output = stderr.read().decode()
            if err_output:
                print(f"Error: {err_output}")
            raise RuntimeError("CVL setup failed")

        print("✅ CVL installed successfully")

    def _build_command(
        self,
        example: str,
        preset: str,
        args: List[str],
        timeout_minutes: Optional[int],
        timeout_action: str
    ) -> str:
        """
        Build shell command with proper escaping and timeout handling.

        Returns:
            Safe shell command string
        """
        # Safely escape all arguments
        safe_example = shlex.quote(example)
        safe_preset = shlex.quote(preset)
        safe_args = [shlex.quote(arg) for arg in args]

        cvl_cmd = f"cvl run {safe_example} {safe_preset} {' '.join(safe_args)}"

        if timeout_minutes:
            # Wrap with timeout that only triggers action on timeout (exit code 124)
            full_cmd = f"""
                set -euo pipefail

                timeout {timeout_minutes}m {cvl_cmd}
                exit_code=$?

                if [ $exit_code -eq 124 ]; then
                    echo "⏱️ Timeout reached after {timeout_minutes} minutes"
                    {timeout_action}
                fi

                exit $exit_code
            """
        else:
            full_cmd = cvl_cmd

        return full_cmd
