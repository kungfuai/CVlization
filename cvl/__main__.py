"""CVL CLI - Command-line interface for CVlization examples."""
import sys
import argparse

from cvl.core.discovery import find_all_examples
from cvl.commands.list import list_examples


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cvl",
        description="CVlization CLI - Discover and run ML examples"
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


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
