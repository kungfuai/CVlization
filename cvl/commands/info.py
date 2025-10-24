"""Info command - display detailed information about an example."""
from typing import List, Dict, Optional, Tuple


def find_matching_examples(examples: List[Dict], identifier: str) -> Tuple[List[Dict], List[str]]:
    """Find examples matching identifier (exact or suffix match).

    Supports flexible matching:
    - Exact: "perception/vision_language/moondream2"
    - Short: "moondream2" matches "perception/vision_language/moondream2"
    - Partial: "line_detection/torch" matches "perception/line_detection/torch"
    - Fuzzy: "moondrem" suggests "moondream2" if no exact/suffix matches

    Args:
        examples: List of example metadata dicts
        identifier: Example identifier (full path, partial path, or short name)

    Returns:
        Tuple of (matching_examples, suggestions)
        - matching_examples: List of matching examples (empty if none found)
        - suggestions: List of suggested paths for fuzzy matching (only when no matches)
    """
    from difflib import get_close_matches

    normalized = identifier.removeprefix("examples/").rstrip("/")
    matches = []
    all_paths = []

    for example in examples:
        path = example.get("_path", "").removeprefix("examples/").rstrip("/")
        all_paths.append(path)

        # Exact match - return immediately
        if path == normalized:
            return [example], []

        # Suffix match - path ends with identifier
        if path.endswith("/" + normalized):
            matches.append(example)

    if matches:
        return matches, []

    # No matches - try fuzzy matching on both full paths and just the last component
    suggestions = get_close_matches(normalized, all_paths, n=5, cutoff=0.5)

    # Also try matching just the example name (last component)
    if not suggestions:
        example_names = {path.split("/")[-1]: path for path in all_paths}
        name_matches = get_close_matches(normalized, example_names.keys(), n=5, cutoff=0.5)
        suggestions = [example_names[name] for name in name_matches]

    return [], suggestions


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
    example_identifier: str
) -> Optional[str]:
    """Get formatted info for an example.

    Args:
        examples: All examples from discovery
        example_identifier: Example identifier (full path, partial path, or short name)

    Returns:
        Formatted info string, or error message if not found/ambiguous
    """
    matches, suggestions = find_matching_examples(examples, example_identifier)

    if len(matches) == 0:
        error_msg = f"✗ Example '{example_identifier}' not found"
        if suggestions:
            error_msg += "\n\nDid you mean:"
            for suggestion in suggestions:
                error_msg += f"\n  • {suggestion}"
        return error_msg
    elif len(matches) > 1:
        # Ambiguous - show all matches
        paths = "\n  • ".join([ex.get("_path", "").removeprefix("examples/") for ex in matches])
        return f"✗ Multiple examples found for '{example_identifier}':\n  • {paths}\n\nUse a more specific path to disambiguate."

    # Single match found
    example = matches[0]
    return format_info(example)
