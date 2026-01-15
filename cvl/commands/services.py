"""Services command - manage deployed serverless services."""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error


def list_services(platform: str = "cerebrium") -> int:
    """List deployed services.

    Args:
        platform: Platform to list services from (default: cerebrium)

    Returns:
        Exit code (0 for success)
    """
    if platform == "cerebrium":
        return _list_cerebrium()
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        return 1


def _list_cerebrium() -> int:
    """List Cerebrium apps."""
    result = subprocess.run(
        ["cerebrium", "apps", "list"],
        capture_output=False,
    )
    return result.returncode


def invoke_service(
    service_name: str,
    platform: str = "cerebrium",
    output_file: Optional[str] = None,
    **kwargs,
) -> int:
    """Invoke a deployed service.

    Args:
        service_name: Name of the service to invoke
        platform: Platform (default: cerebrium)
        output_file: Optional file to save output (e.g., video.mp4)
        **kwargs: Arguments to pass to the service

    Returns:
        Exit code (0 for success)
    """
    if platform == "cerebrium":
        return _invoke_cerebrium(service_name, output_file, **kwargs)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        return 1


def _invoke_cerebrium(
    service_name: str,
    output_file: Optional[str] = None,
    **kwargs,
) -> int:
    """Invoke a Cerebrium service."""
    # Get API key
    api_key = os.environ.get("CEREBRIUM_API_KEY")
    if not api_key:
        print("Error: CEREBRIUM_API_KEY environment variable not set", file=sys.stderr)
        print("\nTo get your API key:", file=sys.stderr)
        print("  1. Go to https://dashboard.cerebrium.ai", file=sys.stderr)
        print("  2. Navigate to your project settings", file=sys.stderr)
        print("  3. Copy the API key", file=sys.stderr)
        print("  4. export CEREBRIUM_API_KEY='your-key'", file=sys.stderr)
        return 1

    # Get project ID from cerebrium config or environment
    project_id = os.environ.get("CEREBRIUM_PROJECT_ID")
    if not project_id:
        # Try to get from cerebrium config
        config_path = Path.home() / ".cerebrium" / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                project_id = config.get("project_id") or config.get("project", {}).get("id")
            except Exception:
                pass

    if not project_id:
        # Try to extract from apps list
        result = subprocess.run(
            ["cerebrium", "apps", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Parse first app ID to extract project ID (format: p-XXXXXXXX-appname)
            for line in result.stdout.splitlines()[1:]:  # Skip header
                parts = line.split()
                if parts:
                    app_id = parts[0]
                    # Extract project ID (p-XXXXXXXX)
                    if app_id.startswith("p-"):
                        project_id = "-".join(app_id.split("-")[:2])
                        break

    if not project_id:
        print("Error: Could not determine Cerebrium project ID", file=sys.stderr)
        print("Set CEREBRIUM_PROJECT_ID environment variable", file=sys.stderr)
        return 1

    # Build endpoint URL
    # Format: https://api.aws.us-east-1.cerebrium.ai/v4/{project_id}/{app_name}/run
    region = os.environ.get("CEREBRIUM_REGION", "us-east-1")
    endpoint = f"https://api.aws.{region}.cerebrium.ai/v4/{project_id}/{service_name}/run"

    print(f"Invoking: {endpoint}", flush=True)
    print(f"Payload: {json.dumps(kwargs, indent=2)}", flush=True)

    # Make request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = json.dumps(kwargs).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

    try:
        # Long timeout for cold starts + video generation
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Error: HTTP {e.code} - {e.reason}", file=sys.stderr)
        try:
            error_body = e.read().decode("utf-8")
            print(f"Response: {error_body}", file=sys.stderr)
        except Exception:
            pass
        return 1
    except urllib.error.URLError as e:
        print(f"Error: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Handle response
    if "error" in result:
        print(f"Service error: {result['error']}", file=sys.stderr)
        return 1

    # Check for base64-encoded output (video, image, etc.)
    output_saved = False
    for key in ["video_base64", "image_base64", "audio_base64"]:
        if key in result:
            if output_file:
                # Decode and save
                content = base64.b64decode(result[key])
                with open(output_file, "wb") as f:
                    f.write(content)
                print(f"Output saved to: {output_file}")
                output_saved = True
                # Remove base64 from printed result
                result[key] = f"<{len(content)} bytes, saved to {output_file}>"
            else:
                # Just indicate size
                content = base64.b64decode(result[key])
                result[key] = f"<{len(content)} bytes, use --output to save>"

    # Print result
    print(f"\nResponse:")
    print(json.dumps(result, indent=2))

    return 0


def service_logs(
    service_name: str,
    platform: str = "cerebrium",
    follow: bool = False,
) -> int:
    """View logs for a service.

    Args:
        service_name: Name of the service
        platform: Platform (default: cerebrium)
        follow: Whether to follow/stream logs

    Returns:
        Exit code (0 for success)
    """
    if platform == "cerebrium":
        return _logs_cerebrium(service_name, follow)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        return 1


def _logs_cerebrium(service_name: str, follow: bool) -> int:
    """View Cerebrium app logs."""
    cmd = ["cerebrium", "logs", service_name]
    if not follow:
        cmd.append("--no-follow")

    result = subprocess.run(cmd)
    return result.returncode


def delete_service(
    service_name: str,
    platform: str = "cerebrium",
    force: bool = False,
) -> int:
    """Delete a deployed service.

    Args:
        service_name: Name of the service to delete
        platform: Platform (default: cerebrium)
        force: Skip confirmation prompt

    Returns:
        Exit code (0 for success)
    """
    if platform == "cerebrium":
        return _delete_cerebrium(service_name, force)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        return 1


def _delete_cerebrium(service_name: str, force: bool) -> int:
    """Delete a Cerebrium app."""
    # First, find the full app ID
    result = subprocess.run(
        ["cerebrium", "apps", "list"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error: Failed to list apps", file=sys.stderr)
        return 1

    # Find app ID matching service name
    app_id = None
    for line in result.stdout.splitlines()[1:]:  # Skip header
        parts = line.split()
        if parts and parts[0].endswith(f"-{service_name}"):
            app_id = parts[0]
            break

    if not app_id:
        print(f"Error: Service '{service_name}' not found", file=sys.stderr)
        return 1

    # Confirm deletion
    if not force:
        print(f"About to delete: {app_id}")
        try:
            response = input("Are you sure? [y/N]: ").strip().lower()
        except EOFError:
            response = "n"

        if response != "y":
            print("Cancelled")
            return 0

    # Delete
    result = subprocess.run(["cerebrium", "apps", "delete", app_id])
    return result.returncode


def service_status(
    service_name: str,
    platform: str = "cerebrium",
) -> int:
    """Check status of a service.

    Args:
        service_name: Name of the service
        platform: Platform (default: cerebrium)

    Returns:
        Exit code (0 for success)
    """
    if platform == "cerebrium":
        return _status_cerebrium(service_name)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        return 1


def _status_cerebrium(service_name: str) -> int:
    """Check Cerebrium app status."""
    result = subprocess.run(
        ["cerebrium", "apps", "list"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Error: Failed to list apps", file=sys.stderr)
        return 1

    # Find app matching service name
    found = False
    for line in result.stdout.splitlines():
        if f"-{service_name}" in line or line.startswith("ID"):
            print(line)
            if f"-{service_name}" in line:
                found = True

    if not found:
        print(f"\nService '{service_name}' not found", file=sys.stderr)
        return 1

    return 0
