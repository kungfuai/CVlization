"""CVL CLI - Command-line interface for CVlization examples."""
import sys
import argparse

from cvl._version import get_full_version
from cvl.core.discovery import find_all_examples
from cvl.commands.list import list_examples
from cvl.commands.info import get_example_info
from cvl.commands.run import run_example
from cvl.commands.export import export_example
from cvl.commands.jobs import list_jobs, tail_logs, kill_job
from cvl.commands.deploy import deploy_example
from cvl.commands.services import (
    list_services,
    invoke_service,
    service_logs,
    service_status,
    delete_service,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cvl",
        description="CVlization CLI - Discover and run ML examples"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=get_full_version()
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List available examples and benchmarks"
    )
    list_parser.add_argument(
        "--capability",
        help="Filter by capability (e.g., perception, generative)"
    )
    list_parser.add_argument(
        "--tag",
        help="Filter by tag (e.g., ocr, video)"
    )
    list_parser.add_argument(
        "--stability",
        help="Filter by stability (stable, beta, experimental)"
    )
    list_parser.add_argument(
        "--keyword",
        "-k",
        help="Search by keyword in name, description, or tags"
    )
    list_parser.add_argument(
        "--format",
        "-f",
        choices=["list", "table"],
        default="list",
        help="Output format (default: list)"
    )
    # Type filter flags (mutually exclusive)
    type_group = list_parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--examples",
        action="store_true",
        help="Show only examples (exclude benchmarks)"
    )
    type_group.add_argument(
        "--benchmarks",
        action="store_true",
        help="Show only benchmarks (exclude examples)"
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about an example"
    )
    info_parser.add_argument(
        "example",
        help="Example path (e.g., generative/minisora)"
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run an example with a specific preset"
    )
    run_parser.add_argument(
        "example",
        help="Example path (e.g., generative/video_generation/minisora)"
    )
    run_parser.add_argument(
        "preset",
        help="Preset to run (e.g., train, predict)"
    )
    run_parser.add_argument(
        "-w", "--work-dir",
        type=str,
        default=None,  # Will be set to cwd in run.py if None
        help="Working directory for inputs/outputs (default: current directory)"
    )
    run_parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live status display (simpler output)"
    )
    run_parser.add_argument(
        "--simple-display",
        action="store_true",
        help="Disable live panel and progress bars (sets HF_HUB_DISABLE_PROGRESS_BARS=1, TQDM_DISABLE=1, PIP_PROGRESS_BAR=off)"
    )
    run_parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run without Docker (requires local Python environment with dependencies)"
    )
    # Remote runner options
    run_parser.add_argument(
        "--runner",
        choices=["local", "sagemaker", "ssh", "k8s", "skypilot"],
        default="local",
        help="Runner to use (default: local Docker)"
    )
    run_parser.add_argument(
        "--instance-type",
        help="Instance type for cloud runners (e.g., ml.g5.xlarge for SageMaker)"
    )
    run_parser.add_argument(
        "--output-path",
        help="S3 path for outputs (required for SageMaker, e.g., s3://bucket/outputs)"
    )
    run_parser.add_argument(
        "--input-path",
        help="S3 path for input data (optional for SageMaker)"
    )
    run_parser.add_argument(
        "--spot",
        action="store_true",
        help="Use spot/preemptible instances for cost savings"
    )
    run_parser.add_argument(
        "--region",
        help="Cloud region (e.g., us-east-2 for AWS)"
    )
    run_parser.add_argument(
        "--max-run-minutes",
        type=int,
        help="Maximum runtime in minutes (default: 60)"
    )
    run_parser.add_argument(
        "--role-arn",
        help="IAM role ARN for SageMaker (or set SAGEMAKER_ROLE_ARN env var)"
    )
    run_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,  # Captures all remaining args without parsing
        # Note: Using REMAINDER allows `cvl run example preset --flag value` without needing `--` separator
        # To require explicit `--` separator (standard Unix convention), change to: nargs="*"
        help="Additional arguments to pass to the script"
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export an example along with the cvlization package for standalone use"
    )
    export_parser.add_argument(
        "example",
        help="Example path (e.g., generative/minisora)"
    )
    export_parser.add_argument(
        "-o", "--dest",
        help="Destination directory for the export (default: ./<example-name>-export)"
    )
    export_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it already exists"
    )

    # deploy command
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy an example to a serverless platform"
    )
    deploy_parser.add_argument(
        "example",
        help="Example name or path to deploy (e.g., ltx2, stable-diffusion)"
    )
    deploy_parser.add_argument(
        "--platform",
        choices=["cerebrium"],
        default="cerebrium",
        help="Target platform (default: cerebrium)"
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare deployment files but don't actually deploy"
    )
    deploy_parser.add_argument(
        "--deploy-dir",
        help="Directory to create deployment files (default: /tmp/cvl-deploy-<name>)"
    )
    deploy_parser.add_argument(
        "--gpu",
        choices=["T4", "L4", "A10", "L40", "A100", "A100_80GB", "H100"],
        help="GPU type override (default: auto-selected based on VRAM requirements)"
    )
    deploy_parser.add_argument(
        "--project",
        help="Cerebrium project ID"
    )
    deploy_parser.add_argument(
        "--skip-model-upload",
        action="store_true",
        help="Skip uploading models to Cerebrium persistent storage"
    )

    # jobs command with subcommands
    jobs_parser = subparsers.add_parser(
        "jobs",
        help="Manage running and recent jobs"
    )
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command", help="Job commands")

    # Common arguments for all jobs subcommands
    def add_common_args(parser):
        parser.add_argument(
            "--runner",
            choices=["sagemaker", "k8s", "skypilot"],
            help="Runner type"
        )
        parser.add_argument(
            "--region",
            help="AWS region for SageMaker"
        )

    # cvl jobs list (also default when no subcommand)
    jobs_list_parser = jobs_subparsers.add_parser(
        "list",
        help="List running and recent jobs"
    )
    add_common_args(jobs_list_parser)
    jobs_list_parser.add_argument(
        "--status",
        choices=["running", "completed", "failed", "stopped"],
        help="Filter by status"
    )
    jobs_list_parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)"
    )

    # cvl jobs logs <job_id>
    jobs_logs_parser = jobs_subparsers.add_parser(
        "logs",
        help="Tail logs for a job"
    )
    jobs_logs_parser.add_argument(
        "job_id",
        help="Job ID to tail logs for"
    )
    add_common_args(jobs_logs_parser)
    jobs_logs_parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Don't follow logs, just print current output"
    )

    # cvl jobs kill <job_id>
    jobs_kill_parser = jobs_subparsers.add_parser(
        "kill",
        help="Stop a running job"
    )
    jobs_kill_parser.add_argument(
        "job_id",
        help="Job ID to stop"
    )
    add_common_args(jobs_kill_parser)

    # Add common args to main jobs parser for "cvl jobs" (defaults to list)
    add_common_args(jobs_parser)
    jobs_parser.add_argument(
        "--status",
        choices=["running", "completed", "failed", "stopped"],
        help="Filter by status"
    )
    jobs_parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)"
    )

    # services command with subcommands
    services_parser = subparsers.add_parser(
        "services",
        help="Manage deployed serverless services"
    )
    services_subparsers = services_parser.add_subparsers(dest="services_command", help="Service commands")

    # Common arguments for services subcommands
    def add_services_common_args(parser):
        parser.add_argument(
            "--platform",
            choices=["cerebrium"],
            default="cerebrium",
            help="Platform (default: cerebrium)"
        )

    # cvl services list
    services_list_parser = services_subparsers.add_parser(
        "list",
        help="List deployed services"
    )
    add_services_common_args(services_list_parser)

    # cvl services invoke <name>
    services_invoke_parser = services_subparsers.add_parser(
        "invoke",
        help="Invoke a deployed service"
    )
    services_invoke_parser.add_argument(
        "name",
        help="Service name to invoke"
    )
    services_invoke_parser.add_argument(
        "--prompt",
        help="Text prompt (for T2V/I2V models)"
    )
    services_invoke_parser.add_argument(
        "--image",
        help="Path to input image (for I2V models)"
    )
    services_invoke_parser.add_argument(
        "--output", "-o",
        help="Output file path (e.g., video.mp4)"
    )
    services_invoke_parser.add_argument(
        "--height",
        type=int,
        help="Video height"
    )
    services_invoke_parser.add_argument(
        "--width",
        type=int,
        help="Video width"
    )
    services_invoke_parser.add_argument(
        "--num-frames",
        type=int,
        help="Number of frames"
    )
    services_invoke_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    add_services_common_args(services_invoke_parser)

    # cvl services logs <name>
    services_logs_parser = services_subparsers.add_parser(
        "logs",
        help="View logs for a service"
    )
    services_logs_parser.add_argument(
        "name",
        help="Service name"
    )
    services_logs_parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Follow/stream logs"
    )
    add_services_common_args(services_logs_parser)

    # cvl services status <name>
    services_status_parser = services_subparsers.add_parser(
        "status",
        help="Check status of a service"
    )
    services_status_parser.add_argument(
        "name",
        help="Service name"
    )
    add_services_common_args(services_status_parser)

    # cvl services delete <name>
    services_delete_parser = services_subparsers.add_parser(
        "delete",
        help="Delete a deployed service"
    )
    services_delete_parser.add_argument(
        "name",
        help="Service name to delete"
    )
    services_delete_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    add_services_common_args(services_delete_parser)

    # Add common args to main services parser (defaults to list)
    add_services_common_args(services_parser)

    return parser


def cmd_list(args) -> int:
    """Handle the list command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        examples = find_all_examples()

        # Determine example_type filter from flags
        example_type = None
        if getattr(args, 'examples', False):
            example_type = 'example'
        elif getattr(args, 'benchmarks', False):
            example_type = 'benchmark'

        output = list_examples(
            examples,
            capability=args.capability,
            tag=args.tag,
            stability=args.stability,
            keyword=getattr(args, 'keyword', None),
            format_type=getattr(args, 'format', 'list'),
            example_type=example_type,
        )
        print(output)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_export(args) -> int:
    """Handle the export command."""
    try:
        examples = find_all_examples()
        exit_code, message = export_example(
            examples,
            args.example,
            dest=getattr(args, "dest", None),
            overwrite=getattr(args, "force", False),
        )
        stream = sys.stderr if exit_code else sys.stdout
        if message:
            print(message, file=stream)
        return exit_code
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_deploy(args) -> int:
    """Handle the deploy command."""
    try:
        return deploy_example(
            example_query=args.example,
            platform=getattr(args, "platform", "cerebrium"),
            dry_run=getattr(args, "dry_run", False),
            deploy_dir=getattr(args, "deploy_dir", None),
            gpu_override=getattr(args, "gpu", None),
            project_id=getattr(args, "project", None),
            skip_model_upload=getattr(args, "skip_model_upload", False),
        )
    except KeyboardInterrupt:
        print("\nDeployment cancelled.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_info(args) -> int:
    """Handle the info command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        examples = find_all_examples()
        output = get_example_info(examples, args.example)

        if output is None:
            print(f"Error: Example '{args.example}' not found", file=sys.stderr)
            return 1

        print(output)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def _extract_runner_args(args):
    """Extract runner-related flags from extra_args due to argparse REMAINDER behavior.

    When using REMAINDER, flags after positional args get captured in extra_args.
    This function extracts known runner flags and updates the args namespace.
    """
    extra = list(getattr(args, 'extra_args', []) or [])
    runner_flags = {
        '--runner': 'runner',
        '--instance-type': 'instance_type',
        '--output-path': 'output_path',
        '--input-path': 'input_path',
        '--region': 'region',
        '--max-run-minutes': 'max_run_minutes',
        '--role-arn': 'role_arn',
    }
    bool_flags = {'--spot': 'spot'}

    new_extra = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg in runner_flags:
            # Flag with value
            if i + 1 < len(extra):
                attr = runner_flags[arg]
                value = extra[i + 1]
                # Convert types as needed
                if attr == 'max_run_minutes':
                    value = int(value)
                setattr(args, attr, value)
                i += 2
            else:
                new_extra.append(arg)
                i += 1
        elif arg in bool_flags:
            setattr(args, bool_flags[arg], True)
            i += 1
        else:
            new_extra.append(arg)
            i += 1

    args.extra_args = new_extra
    return args


def cmd_run(args) -> int:
    """Handle the run command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Extract runner flags that might be in extra_args due to REMAINDER
    args = _extract_runner_args(args)
    runner = getattr(args, 'runner', 'local')

    # Dispatch to remote runner if specified
    if runner and runner != "local":
        return cmd_run_remote(args, runner)

    # Local Docker/native execution
    try:
        extra_args = list(getattr(args, 'extra_args', [])) or []
        if getattr(args, "simple_display", False):
            extra_args = [a for a in extra_args if a != "--simple-display"]

        examples = find_all_examples()
        exit_code, error_msg = run_example(
            examples,
            args.example,
            args.preset,
            extra_args,
            work_dir=getattr(args, 'work_dir', None),
            no_live=getattr(args, 'no_live', False) or getattr(args, 'simple_display', False),
            simple_display=getattr(args, 'simple_display', False),
            no_docker=getattr(args, 'no_docker', False)
        )

        if error_msg:
            print(f"Error: {error_msg}", file=sys.stderr)

        return exit_code
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_run_remote(args, runner: str) -> int:
    """Handle remote runner execution.

    Args:
        args: Parsed command line arguments
        runner: Runner name (sagemaker, ssh, k8s, skypilot)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    import os
    from pathlib import Path
    from cvl.core.config import get_runner_config, validate_sagemaker_config
    from cvl.core.matching import find_matching_examples
    from cvl.core.discovery import find_repo_root

    # Find the example
    examples = find_all_examples()
    matches, _ = find_matching_examples(examples, args.example)

    if not matches:
        print(f"Error: Example '{args.example}' not found", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"Error: Ambiguous example '{args.example}'. Matches: {[m['name'] for m in matches]}", file=sys.stderr)
        return 1

    example = matches[0]
    repo_root = find_repo_root()
    example_path = repo_root / example["_path"]

    # Build CLI overrides from args
    cli_overrides = {
        "instance_type": getattr(args, 'instance_type', None),
        "output_path": getattr(args, 'output_path', None),
        "input_path": getattr(args, 'input_path', None),
        "spot": getattr(args, 'spot', False) or None,  # Only set if True
        "region": getattr(args, 'region', None),
        "max_run_minutes": getattr(args, 'max_run_minutes', None),
        "role_arn": getattr(args, 'role_arn', None),
    }

    # Get merged config
    config = get_runner_config(runner, example_path, cli_overrides)

    # Get extra args
    extra_args = list(getattr(args, 'extra_args', [])) or []

    if runner == "sagemaker":
        return _run_sagemaker(args.example, args.preset, extra_args, config)
    elif runner == "ssh":
        print("SSH runner not yet integrated with cvl run", file=sys.stderr)
        return 1
    elif runner == "k8s":
        print("K8s runner not yet integrated with cvl run", file=sys.stderr)
        return 1
    elif runner == "skypilot":
        print("SkyPilot runner not yet integrated with cvl run", file=sys.stderr)
        return 1
    else:
        print(f"Unknown runner: {runner}", file=sys.stderr)
        return 1


def _run_sagemaker(example: str, preset: str, extra_args: list, config: dict) -> int:
    """Run example on SageMaker.

    Args:
        example: Example name/path
        preset: Preset to run
        extra_args: Additional arguments for the script
        config: Merged runner config

    Returns:
        Exit code
    """
    import os
    from cvl.core.config import validate_sagemaker_config

    # Validate config
    error = validate_sagemaker_config(config)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        from cvl.runners import SageMakerRunner
    except ImportError:
        print("boto3 not installed. Install with: pip install boto3", file=sys.stderr)
        return 1

    # Get role ARN (from config or env)
    role_arn = config.get("role_arn") or os.environ.get("SAGEMAKER_ROLE_ARN")
    region = config.get("region") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    runner = SageMakerRunner(role_arn=role_arn, region=region)

    return runner.run(
        example=example,
        preset=preset,
        args=extra_args,
        instance_type=config.get("instance_type", "ml.g5.xlarge"),
        output_path=config["output_path"],
        input_path=config.get("input_path"),
        instance_count=config.get("instance_count", 1),
        spot=config.get("spot", False),
        max_run_minutes=config.get("max_run_minutes", 60),
        volume_size_gb=config.get("volume_size_gb", 50),
        download_outputs=config.get("download_outputs", True),
    )


def cmd_jobs(args) -> int:
    """Handle the jobs command and subcommands."""
    jobs_cmd = getattr(args, 'jobs_command', None)

    if jobs_cmd is None or jobs_cmd == "list":
        # Default: list jobs
        return list_jobs(
            runner=getattr(args, 'runner', None),
            status=getattr(args, 'status', None),
            max_results=getattr(args, 'max_results', 20),
            region=getattr(args, 'region', None),
        )
    elif jobs_cmd == "logs":
        return tail_logs(
            job_id=args.job_id,
            runner=getattr(args, 'runner', None),
            follow=not getattr(args, 'no_follow', False),
            region=getattr(args, 'region', None),
        )
    elif jobs_cmd == "kill":
        return kill_job(
            job_id=args.job_id,
            runner=getattr(args, 'runner', None),
            region=getattr(args, 'region', None),
        )
    else:
        print(f"Unknown jobs subcommand: {jobs_cmd}")
        return 1


def cmd_services(args) -> int:
    """Handle the services command and subcommands."""
    import base64

    services_cmd = getattr(args, 'services_command', None)
    platform = getattr(args, 'platform', 'cerebrium')

    if services_cmd is None or services_cmd == "list":
        return list_services(platform=platform)

    elif services_cmd == "invoke":
        # Build kwargs from args
        kwargs = {}
        if args.prompt:
            kwargs["prompt"] = args.prompt
        if args.height:
            kwargs["height"] = args.height
        if args.width:
            kwargs["width"] = args.width
        if getattr(args, 'num_frames', None):
            kwargs["num_frames"] = args.num_frames
        if args.seed:
            kwargs["seed"] = args.seed

        # Handle image input (read file and base64 encode)
        if args.image:
            try:
                with open(args.image, "rb") as f:
                    image_bytes = f.read()
                kwargs["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            except FileNotFoundError:
                print(f"Error: Image file not found: {args.image}", file=sys.stderr)
                return 1

        if not kwargs.get("prompt"):
            print("Error: --prompt is required", file=sys.stderr)
            return 1

        return invoke_service(
            service_name=args.name,
            platform=platform,
            output_file=getattr(args, 'output', None),
            **kwargs,
        )

    elif services_cmd == "logs":
        return service_logs(
            service_name=args.name,
            platform=platform,
            follow=getattr(args, 'follow', False),
        )

    elif services_cmd == "status":
        return service_status(
            service_name=args.name,
            platform=platform,
        )

    elif services_cmd == "delete":
        return delete_service(
            service_name=args.name,
            platform=platform,
            force=getattr(args, 'force', False),
        )

    else:
        print(f"Unknown services subcommand: {services_cmd}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "deploy":
        return cmd_deploy(args)
    elif args.command == "jobs":
        return cmd_jobs(args)
    elif args.command == "services":
        return cmd_services(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
