"""Info command - display detailed information about an example."""
from typing import List, Dict, Optional


def find_example(examples: List[Dict], example_path: str) -> Optional[Dict]:
    """Find an example by its path.

    Args:
        examples: List of example metadata dicts
        example_path: Path to example (e.g., "generative/minisora" or "examples/generative/minisora")

    Returns:
        Example dict if found, None otherwise
    """
    # Normalize path - remove leading "examples/" if present
    normalized_path = example_path.removeprefix("examples/").rstrip("/")

    for example in examples:
        example_rel_path = example.get("_path", "").removeprefix("examples/").rstrip("/")
        if example_rel_path == normalized_path:
            return example

    return None


def format_info(example: Dict) -> str:
    """Format example metadata as readable text.

    Args:
        example: Example metadata dict

    Returns:
        Formatted info string
    """
    lines = []

    # Basic info
    lines.append(f"Name: {example.get('name', 'unknown')}")
    lines.append(f"Path: {example.get('_path', 'unknown')}")
    lines.append(f"Capability: {example.get('capability', 'unknown')}")
    lines.append(f"Stability: {example.get('stability', 'unknown')}")

    # Tags
    tags = example.get('tags', [])
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")

    # Description
    description = example.get('description')
    if description:
        lines.append(f"\nDescription:")
        lines.append(f"  {description}")

    # Resources
    resources = example.get('resources', {})
    if resources:
        lines.append("\nResources:")
        if 'gpu' in resources:
            lines.append(f"  GPU: {resources['gpu']}")
        if 'vram' in resources:
            lines.append(f"  VRAM: {resources['vram']}")
        if 'training_time' in resources:
            lines.append(f"  Training time: {resources['training_time']}")

    # Datasets
    datasets = example.get('datasets', [])
    if datasets:
        lines.append("\nDatasets:")
        for dataset in datasets:
            if isinstance(dataset, dict):
                name = dataset.get('name', 'unknown')
                lines.append(f"  - {name}")
            else:
                lines.append(f"  - {dataset}")

    # Presets
    presets = example.get('presets', [])
    if presets:
        lines.append("\nPresets:")
        if isinstance(presets, list):
            for preset in presets:
                lines.append(f"  - {preset}")
        elif isinstance(presets, dict):
            for preset_name, preset_info in presets.items():
                if isinstance(preset_info, dict):
                    script = preset_info.get('script', f'{preset_name}.sh')
                    desc = preset_info.get('description', preset_name)
                    lines.append(f"  - {preset_name}: {desc}")
                    lines.append(f"    Script: {script}")
                else:
                    lines.append(f"  - {preset_name}")

    return "\n".join(lines)


def get_example_info(
    examples: List[Dict],
    example_path: str
) -> Optional[str]:
    """Get formatted info for an example.

    Args:
        examples: All examples from discovery
        example_path: Path to the example

    Returns:
        Formatted info string, or None if example not found
    """
    example = find_example(examples, example_path)
    if example is None:
        return None

    return format_info(example)
