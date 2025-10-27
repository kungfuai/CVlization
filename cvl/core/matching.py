"""Example matching utilities for CVL commands."""
from typing import List, Dict, Tuple
from difflib import get_close_matches


def find_matching_examples(examples: List[Dict], identifier: str) -> Tuple[List[Dict], List[str]]:
    """Find examples matching identifier (exact or suffix match).

    Supports flexible matching:
    - Exact: "perception/vision_language/moondream2"
    - Short: "moondream2" matches "perception/vision_language/moondream2"
    - Partial: "line_detection/torch" matches "perception/line_detection/torch"
    - Name: "panoptic-segmentation-mmdet" matches example with that name field
    - Fuzzy: "moondrem" suggests "moondream2" if no exact/suffix matches

    Args:
        examples: List of example metadata dicts
        identifier: Example identifier (full path, partial path, short name, or name field)

    Returns:
        Tuple of (matching_examples, suggestions)
        - matching_examples: List of matching examples (empty if none found)
        - suggestions: List of suggested paths for fuzzy matching (only when no matches)
    """
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

    # Try matching by name field (from example.yaml)
    for example in examples:
        name = example.get("name", "")
        if name == normalized:
            return [example], []

    # No matches - try fuzzy matching on both full paths and just the last component
    suggestions = get_close_matches(normalized, all_paths, n=5, cutoff=0.5)

    # Also try matching just the example name (last component)
    if not suggestions:
        example_names = {path.split("/")[-1]: path for path in all_paths}
        name_matches = get_close_matches(normalized, example_names.keys(), n=5, cutoff=0.5)
        suggestions = [example_names[name] for name in name_matches]

    return [], suggestions
