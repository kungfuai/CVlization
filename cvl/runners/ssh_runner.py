"""SSH-based remote execution for CVL examples."""

import shlex
import socket
import sys
import threading
import time
from typing import Optional, List
from pathlib import Path


class ChannelWatchdog:
    """
    Monitors SSH channel activity and closes it if idle too long.

    This handles the case where the remote process is alive but not producing
    output (e.g., stuck in a loop, waiting on I/O). The socket timeout only
    catches network-level issues, not application-level hangs.
    """

    def __init__(self, channel, timeout_seconds: int):
        self.channel = channel
        self.timeout = timeout_seconds
        self.last_activity = time.time()
        self.stopped = False
        self.timed_out = False
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        """Start the watchdog thread."""
        self._thread.start()

    def stop(self):
        """Stop the watchdog thread."""
        self.stopped = True

    def ping(self):
        """Call this when activity occurs to reset the timeout."""
        self.last_activity = time.time()

    def _watch(self):
        """Background thread that monitors for inactivity."""
        while not self.stopped and not self.channel.closed:
            idle_time = time.time() - self.last_activity
            if idle_time > self.timeout:
                self.timed_out = True
                self.channel.close()
                break
            # Check every 10 seconds or 1/10th of timeout, whichever is smaller
            time.sleep(min(10, self.timeout / 10))


class SSHRunner:
    """
    Execute CVL examples on remote hosts via SSH.

    Limitations:
    - No artifact retrieval: Training outputs (checkpoints, logs, models) remain
      on the remote instance. Configure your training script to upload artifacts
      to cloud storage (S3, GCS, W&B, etc.) before completion.

    Security Notes:
    - Uses AutoAddPolicy for host key verification (vulnerable to MITM).
      For production use, consider verifying host keys via cloud provider API.
    - The timeout_action parameter is not escaped; only use trusted values.
    """

    # Socket-level timeout for SSH operations (seconds).
    # Prevents indefinite hangs if network dies mid-command.
    CHANNEL_TIMEOUT = 300  # 5 minutes without data = assume dead

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

            # Execute command with watchdog to prevent indefinite hangs
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            channel = stdout.channel

            # Socket timeout for network-level issues
            channel.settimeout(self.CHANNEL_TIMEOUT)

            # Watchdog for application-level hangs (no output but connection alive)
            watchdog = ChannelWatchdog(channel, self.CHANNEL_TIMEOUT)
            watchdog.start()

            # Stream output in real-time
            try:
                for line in stdout:
                    watchdog.ping()  # Reset timeout on each line
                    print(line, end='')
            except socket.timeout:
                print(f"\n⚠️ No output received for {self.CHANNEL_TIMEOUT}s, assuming connection lost")
                return 1
            except OSError as e:
                # Channel closed by watchdog or network error
                if watchdog.timed_out:
                    print(f"\n⚠️ No output received for {self.CHANNEL_TIMEOUT}s, assuming process hung")
                    return 1
                raise
            finally:
                watchdog.stop()

            # Get exit code
            exit_code = channel.recv_exit_status()

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
            # Note: We use `|| true` to capture the exit code without triggering set -e
            full_cmd = f"""
                set -uo pipefail

                timeout {timeout_minutes}m {cvl_cmd} || exit_code=$?
                exit_code=${{exit_code:-0}}

                if [ $exit_code -eq 124 ]; then
                    echo "⏱️ Timeout reached after {timeout_minutes} minutes"
                    {timeout_action}
                fi

                exit $exit_code
            """
        else:
            full_cmd = cvl_cmd

        return full_cmd
