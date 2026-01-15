"""Deploy command - deploy CVL examples to serverless platforms."""

import os
import sys
from pathlib import Path
from typing import Optional

from cvl.core.discovery import find_all_examples, find_repo_root
from cvl.core.matching import find_matching_examples
from cvl.deployers.cerebrium import CerebriumDeployer, CEREBRIUM_GPUS


def deploy_example(
    example_query: str,
    platform: str = "cerebrium",
    dry_run: bool = False,
    deploy_dir: Optional[str] = None,
    gpu_override: Optional[str] = None,
) -> int:
    """
    Deploy a CVL example to a serverless platform.

    Args:
        example_query: Example name or path to deploy
        platform: Target platform (default: cerebrium)
        dry_run: If True, prepare files but don't deploy
        deploy_dir: Optional directory for deployment files
        gpu_override: Override GPU type (e.g., "A10", "A100", "H100")

    Returns:
        Exit code (0 for success)
    """
    # Find the example
    print(f"Finding example: {example_query}")
    examples = find_all_examples()
    matches, _ = find_matching_examples(examples, example_query)

    if not matches:
        print(f"Error: Example '{example_query}' not found", file=sys.stderr)
        return 1

    if len(matches) > 1:
        print(f"Error: Ambiguous example '{example_query}'.", file=sys.stderr)
        print(f"Matches: {[m['name'] for m in matches]}", file=sys.stderr)
        return 1

    example = matches[0]
    repo_root = find_repo_root()
    example_path = repo_root / example["_path"]

    print(f"  Found: {example['name']}")
    print(f"  Path: {example_path}")

    # Check for predict preset (required for deployment)
    presets = example.get("presets", {})
    if "predict" not in presets and "serve" not in presets:
        print(f"\nError: Example '{example['name']}' has no predict or serve preset.", file=sys.stderr)
        print("Deployment requires an inference entry point.", file=sys.stderr)
        return 1

    preset_type = "serve" if "serve" in presets else "predict"
    print(f"  Preset: {preset_type}")

    # Show hardware requirements
    resources = example.get("resources", {})
    print(f"\nHardware requirements:")
    print(f"  GPU: {resources.get('gpu_count', resources.get('gpu', 1))}")
    print(f"  VRAM: {resources.get('vram_gb', 'not specified')} GB")
    print(f"  Disk: {resources.get('disk_gb', 'not specified')} GB")

    # Platform-specific deployment
    sys.stdout.flush()
    if platform == "cerebrium":
        return _deploy_cerebrium(example_path, example, dry_run, deploy_dir, gpu_override)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        print("Supported platforms: cerebrium", file=sys.stderr)
        return 1


def _deploy_cerebrium(
    example_path: Path,
    example_meta: dict,
    dry_run: bool,
    deploy_dir: Optional[str],
    gpu_override: Optional[str] = None,
) -> int:
    """Deploy to Cerebrium platform."""
    deployer = CerebriumDeployer(example_path, example_meta, gpu_override=gpu_override)

    # Check if example is supported for automatic deployment
    if not deployer.is_supported():
        print(f"\nError: {deployer.get_unsupported_message()}", file=sys.stderr, flush=True)
        return 1

    # Show GPU configuration
    gpu_id, gpu_name, vram_needed = deployer.get_gpu_config()
    gpu_info = CEREBRIUM_GPUS.get(gpu_name, {})
    print(f"\nCerebrium GPU configuration:")
    print(f"  GPU: {gpu_name} ({gpu_id})")
    print(f"  VRAM: {gpu_info.get('vram', 'N/A')} GB")
    print(f"  Plan: {gpu_info.get('plan', 'N/A')}")
    if gpu_override:
        print(f"  (User override: --gpu {gpu_override})")
    else:
        print(f"  (Auto-selected for {vram_needed}GB VRAM requirement)")
    print(f"\nAvailable GPUs: {', '.join(CEREBRIUM_GPUS.keys())}")
    print(f"Override with: cvl deploy {example_meta.get('name')} --gpu <GPU>")

    # Check CLI (skip for dry-run)
    if not dry_run:
        print("\nChecking Cerebrium CLI...")
        ready, message = deployer.check_cli()
        if not ready:
            print(f"  {message}", file=sys.stderr)
            print("\nTo authenticate with Cerebrium:", file=sys.stderr)
            print("  1. Login: cerebrium login", file=sys.stderr)
            print("  2. Or set CEREBRIUM_TOKEN env var for CI/CD", file=sys.stderr)
            return 1
        print(f"  {message}")

    # Prepare deployment
    print("\nPreparing deployment...")
    deploy_path = Path(deploy_dir) if deploy_dir else None
    deploy_path = deployer.prepare_deployment(deploy_path)

    print(f"\nDeployment files ready: {deploy_path}")
    print("Files:")
    for f in sorted(deploy_path.iterdir()):
        if f.is_file():
            print(f"  {f.name}")
        else:
            print(f"  {f.name}/")

    # Deploy
    if dry_run:
        print("\n[Dry run] Skipping actual deployment")
        print(f"[Dry run] To deploy manually, run:")
        print(f"  cd {deploy_path}")
        print(f"  cerebrium deploy")
        return 0

    # Ask for confirmation
    print("\nReady to deploy to Cerebrium.")
    try:
        response = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        response = "n"

    if response != "y":
        print("Deployment cancelled.")
        print(f"Files are still available at: {deploy_path}")
        return 0

    return deployer.deploy(deploy_path, dry_run=False)
