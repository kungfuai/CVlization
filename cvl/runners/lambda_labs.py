"""Lambda Labs Cloud integration for CVL."""

import atexit
import os
import signal
import sys
import time
from typing import Optional, List

import requests

from .ssh_runner import SSHRunner


class LambdaLabsRunner:
    """
    Run CVL examples on Lambda Labs Cloud with automatic instance management.

    Features:
    - Automatic instance creation and termination
    - Cleanup on exit, Ctrl+C, or errors
    - Cost estimation before launch
    - Timeout support via remote shutdown
    """

    API_BASE = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Lambda Labs runner.

        Args:
            api_key: Lambda Labs API key (defaults to LAMBDA_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Lambda Labs API key required. "
                "Set LAMBDA_API_KEY environment variable or pass api_key parameter."
            )

        self.instance_id = None
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # Setup cleanup handlers
        self._setup_cleanup_handlers()

    def run(
        self,
        example: str,
        preset: str,
        args: List[str],
        gpu_type: str = "gpu_1x_a100_sxm4",
        region: str = "us-west-1",
        timeout_minutes: Optional[int] = None
    ) -> int:
        """
        Launch Lambda instance, run CVL example, and terminate.

        Args:
            example: Example name (e.g., "nanogpt")
            preset: Preset name (e.g., "train")
            args: Additional arguments
            gpu_type: GPU instance type
            region: Lambda Labs region
            timeout_minutes: Optional timeout (kills job and shuts down instance)

        Returns:
            Exit code from command
        """
        # Validate timeout
        if timeout_minutes is not None and timeout_minutes <= 0:
            raise ValueError(f"timeout_minutes must be positive, got {timeout_minutes}")

        try:
            # Create instance
            print(f"🚀 Launching {gpu_type} in {region}...")
            self.instance_id = self._create_instance(gpu_type, region, example)
            print(f"✅ Instance created: {self.instance_id}")

            # Wait for instance to be ready
            print(f"⏳ Waiting for instance to be ready...")
            ip = self._wait_for_ready(self.instance_id)
            print(f"✅ Instance ready at {ip}")

            # Run command via SSH
            ssh_runner = SSHRunner()
            exit_code = ssh_runner.run_remote(
                host=f"ubuntu@{ip}",
                example=example,
                preset=preset,
                args=args,
                timeout_minutes=timeout_minutes,
                timeout_action="sudo shutdown -h now",  # Shutdown on timeout
                setup_cvl=True
            )

            return exit_code

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            return 130

        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

        finally:
            # Always terminate instance
            self._terminate()

    def _create_instance(self, gpu_type: str, region: str, name: str) -> str:
        """
        Create Lambda Labs instance.

        Returns:
            instance_id
        """
        response = self.session.post(
            f"{self.API_BASE}/instance-operations/launch",
            json={
                "region_name": region,
                "instance_type_name": gpu_type,
                "name": f"cvl-{name}-{int(time.time())}",
                "quantity": 1
            }
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except (ValueError, KeyError):
                error_msg = f"HTTP {response.status_code}"
            raise RuntimeError(f"Failed to create instance: {error_msg}")

        try:
            data = response.json()["data"]
            instance_ids = data.get("instance_ids", [])

            if not instance_ids:
                raise RuntimeError("No instance ID returned from API")

            return instance_ids[0]
        except (KeyError, ValueError, IndexError) as e:
            raise RuntimeError(f"Invalid API response format: {e}")

    def _wait_for_ready(self, instance_id: str, timeout: int = 300) -> str:
        """
        Wait for instance to be active and SSH-ready.

        Returns:
            IP address of instance
        """
        start = time.time()

        while time.time() - start < timeout:
            # Get instance status
            response = self.session.get(f"{self.API_BASE}/instances/{instance_id}")

            if response.status_code != 200:
                print(f"⚠️ Error checking instance status: {response.status_code}")
                time.sleep(5)
                continue

            try:
                data = response.json()
                instance = data.get("data")
                if not instance:
                    print(f"⚠️ Unexpected API response format")
                    time.sleep(5)
                    continue

                status = instance.get("status")
                ip = instance.get("ip")
            except (KeyError, ValueError) as e:
                print(f"⚠️ Error parsing API response: {e}")
                time.sleep(5)
                continue

            if status == "active" and ip:
                return ip

            elif status in ["booting", "unhealthy"]:
                print(f"   Status: {status}...")

            else:
                print(f"   Unexpected status: {status}")

            time.sleep(5)

        raise TimeoutError(f"Instance not ready after {timeout}s")

    def _terminate(self):
        """Terminate the instance if it exists."""
        if not self.instance_id:
            return

        print(f"🧹 Terminating instance {self.instance_id}...")

        try:
            response = self.session.post(
                f"{self.API_BASE}/instance-operations/terminate",
                json={"instance_ids": [self.instance_id]}
            )

            if response.status_code == 200:
                print("✅ Instance terminated successfully")
            else:
                print(f"⚠️ Termination request returned status {response.status_code}")

        except Exception as e:
            print(f"⚠️ Error terminating instance: {e}")
            print(f"   You may need to manually terminate instance {self.instance_id}")

        finally:
            self.instance_id = None

    def _setup_cleanup_handlers(self):
        """Setup handlers to ensure instance cleanup on exit/interrupt."""
        # Normal exit
        atexit.register(self._terminate)

        # Ctrl+C
        def signal_handler(signum, frame):
            print("\n⚠️ Interrupted! Cleaning up...")
            self._terminate()
            sys.exit(130)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
