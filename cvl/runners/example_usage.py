#!/usr/bin/env python3
"""
Example usage of CVL remote runners.

This shows how to use SSHRunner and LambdaLabsRunner to execute
CVL examples on remote machines.
"""

from cvl.runners import SSHRunner, LambdaLabsRunner


def example_ssh_runner():
    """Example: Run on existing remote host via SSH."""
    print("=== SSH Runner Example ===\n")

    runner = SSHRunner()

    # Run on an existing Lambda Labs (or any) instance
    exit_code = runner.run_remote(
        host="ubuntu@123.45.67.89",  # Your instance IP
        example="nanogpt",
        preset="train",
        args=["--max_iters=100", "--batch_size=16"],
        timeout_minutes=120,  # Kill after 2 hours
        timeout_action="sudo shutdown -h now",  # Shutdown on timeout
        setup_cvl=True  # Auto-install CVL if needed
    )

    print(f"\nExit code: {exit_code}")


def example_lambda_labs_runner():
    """Example: Automated Lambda Labs instance management."""
    print("=== Lambda Labs Runner Example ===\n")

    # Requires LAMBDA_API_KEY environment variable
    runner = LambdaLabsRunner()

    # Automatically creates instance, runs job, and terminates
    exit_code = runner.run(
        example="nanogpt",
        preset="train",
        args=["--max_iters=1000", "--batch_size=16"],
        gpu_type="gpu_1x_a100_sxm4",
        region="us-west-1",
        timeout_minutes=120  # Safety timeout
    )

    print(f"\nExit code: {exit_code}")


def example_lambda_quick():
    """Example: Quick Lambda Labs run with defaults."""
    print("=== Quick Lambda Run ===\n")

    runner = LambdaLabsRunner()

    # Minimal usage - uses sensible defaults
    runner.run(
        example="nanogpt",
        preset="train",
        args=["--max_iters=100"]
    )


if __name__ == "__main__":
    import sys

    print("CVL Remote Runner Examples")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python example_usage.py ssh         # SSH runner example")
    print("  python example_usage.py lambda      # Lambda Labs full example")
    print("  python example_usage.py quick       # Lambda Labs quick example")
    print()

    if len(sys.argv) < 2:
        print("Please specify: ssh, lambda, or quick")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "ssh":
        example_ssh_runner()
    elif mode == "lambda":
        example_lambda_labs_runner()
    elif mode == "quick":
        example_lambda_quick()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
