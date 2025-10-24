"""CVL CLI - Command-line interface for CVlization examples."""
import sys
import argparse

from cvl._version import get_full_version
from cvl.core.discovery import find_all_examples
from cvl.commands.list import list_examples
from cvl.commands.info import get_example_info
from cvl.commands.run import run_example


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
        help="List available examples"
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
        "extra_args",
        nargs=argparse.REMAINDER,  # Captures all remaining args without parsing
        # Note: Using REMAINDER allows `cvl run example preset --flag value` without needing `--` separator
        # To require explicit `--` separator (standard Unix convention), change to: nargs="*"
        help="Additional arguments to pass to the script"
    )

    return parser


def cmd_list(args) -> int:
    """Handle the list command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        examples = find_all_examples()
        output = list_examples(
            examples,
            capability=args.capability,
            tag=args.tag,
            stability=args.stability
        )
        print(output)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
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


def cmd_run(args) -> int:
    """Handle the run command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        examples = find_all_examples()
        exit_code, error_msg = run_example(
            examples,
            args.example,
            args.preset,
            args.extra_args,
            work_dir=getattr(args, 'work_dir', None),
            no_live=getattr(args, 'no_live', False)
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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
