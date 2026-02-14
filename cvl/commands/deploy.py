"""Deploy command - deploy CVL examples to serverless platforms."""

import sys
from pathlib import Path
from typing import Optional

from cvl.core.discovery import find_all_examples, find_repo_root
from cvl.core.matching import find_matching_examples
from cvl.deployers.base import BaseDeployer
from cvl.deployers.cerebrium import CerebriumDeployer
from cvl.deployers.modal import ModalDeployer


def deploy_example(
    example_query: str,
    platform: str = "cerebrium",
    dry_run: bool = False,
    deploy_dir: Optional[str] = None,
    gpu_override: Optional[str] = None,
    project_id: Optional[str] = None,
    skip_model_upload: bool = False,
    only_model_upload: bool = False,
) -> int:
    """Deploy a CVL example to a serverless platform."""
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

    # Create platform-specific deployer
    sys.stdout.flush()
    if platform == "cerebrium":
        deployer = CerebriumDeployer(example_path, example, gpu_override=gpu_override, project_id=project_id)
    elif platform == "modal":
        deployer = ModalDeployer(example_path, example, gpu_override=gpu_override)
    else:
        print(f"Error: Unknown platform '{platform}'", file=sys.stderr)
        print("Supported platforms: cerebrium, modal", file=sys.stderr)
        return 1

    return _run_deploy(deployer, dry_run, deploy_dir, gpu_override, skip_model_upload, only_model_upload)


def _run_deploy(
    deployer: BaseDeployer,
    dry_run: bool,
    deploy_dir: Optional[str],
    gpu_override: Optional[str],
    skip_model_upload: bool,
    only_model_upload: bool,
) -> int:
    """Shared deployment flow for all platforms."""
    platform = deployer.platform_name

    # Check if example is supported
    if not deployer.is_supported():
        print(f"\nError: {deployer.get_unsupported_message()}", file=sys.stderr, flush=True)
        return 1

    # Show GPU configuration
    gpu_id, gpu_name, vram_needed = deployer.get_gpu_config()
    print(f"\n{platform} GPU configuration:")
    print(f"  GPU: {gpu_name} ({gpu_id})")
    for line in deployer.format_gpu_info(gpu_name, gpu_id, vram_needed):
        print(line)
    if gpu_override:
        print(f"  (User override: --gpu {gpu_override})")
    else:
        print(f"  (Auto-selected for {vram_needed}GB VRAM requirement)")
    print(f"\nAvailable GPUs: {', '.join(deployer.GPU_TABLE.keys())}")
    print(f"Override with: cvl deploy {deployer.name} --gpu <GPU>")

    # Check CLI (skip for dry-run)
    if not dry_run:
        print(f"\nChecking {platform} CLI...", flush=True)
        ready, message = deployer.check_cli()
        if not ready:
            print(f"\n{message}", file=sys.stderr, flush=True)
            return 1

    # Handle --only-model-upload
    if only_model_upload:
        print("\n[--only-model-upload] Uploading models only, skipping code deployment")
        models = deployer.get_required_models()
        if not models:
            print("No models to upload for this example")
            return 0

        _print_model_list(models)
        results = deployer.upload_models()
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"\nError: Some models failed to upload: {failed}")
            return 1
        print("\nModel upload complete!")
        return 0

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

    if dry_run:
        print("\n[Dry run] Skipping actual deployment")
        print(f"[Dry run] To deploy manually, run:")
        print(f"  cd {deploy_path}")
        print(f"  {deployer.dry_run_command}")
        return 0

    # Ask for confirmation
    print(f"\nReady to deploy to {platform}.")
    try:
        response = input("Proceed? [y/N]: ").strip().lower()
    except EOFError:
        response = "n"

    if response != "y":
        print("Deployment cancelled.")
        print(f"Files are still available at: {deploy_path}")
        return 0

    # Upload models
    if not skip_model_upload:
        models = deployer.get_required_models()
        if models:
            _print_model_list(models)
            results = deployer.upload_models()
            failed = [k for k, v in results.items() if not v]
            if failed:
                print(f"\nWarning: Some models failed to upload: {failed}")
                print("The deployment will continue, but cold starts may be slow.")
    else:
        print(f"\n[--skip-model-upload] Skipping model upload to {platform} storage")

    return deployer.deploy(deploy_path, dry_run=False)


def _print_model_list(models: dict) -> None:
    """Print formatted model list."""
    model_strs = []
    for repo_id, files in models.items():
        if files:
            model_strs.append(f"{repo_id} ({len(files)} files)")
        else:
            model_strs.append(f"{repo_id} (full)")
    print(f"Models to upload: {', '.join(model_strs)}")
