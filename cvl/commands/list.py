"""List command - display available examples."""
from typing import List, Dict, Optional


def filter_examples(
    examples: List[Dict],
    capability: Optional[str] = None,
    tag: Optional[str] = None,
    stability: Optional[str] = None,
) -> List[Dict]:
    """Filter examples by criteria (pure function).

    Args:
        examples: List of example metadata dicts
        capability: Filter by capability (e.g., "perception", "generative")
        tag: Filter by tag (e.g., "ocr", "video")
        stability: Filter by stability (e.g., "stable", "beta")

    Returns:
        Filtered list of examples
    """
    filtered = examples

    if capability:
        filtered = [e for e in filtered if capability.lower() in e.get('capability', '').lower()]

    if tag:
        filtered = [e for e in filtered if tag.lower() in [t.lower() for t in e.get('tags', [])]]

    if stability:
        filtered = [e for e in filtered if e.get('stability', '').lower() == stability.lower()]

    return filtered


def format_table(examples: List[Dict]) -> str:
    """Format examples as a simple table.

    Args:
        examples: List of example metadata dicts

    Returns:
        Formatted table as string
    """
    if not examples:
        return "No examples found."

    # Calculate column widths
    max_name = max(len(e.get('name', '')) for e in examples)
    max_cap = max(len(e.get('capability', '')) for e in examples)
    max_path = max(len(e.get('_path', '')) for e in examples)

    # Build table
    header = f"{'NAME':<{max_name}}  {'CAPABILITY':<{max_cap}}  {'PATH':<{max_path}}  STABILITY"
    separator = "-" * len(header)

    rows = []
    for e in examples:
        name = e.get('name', 'unknown')
        cap = e.get('capability', '')
        path = e.get('_path', '')
        stability = e.get('stability', 'unknown')
        rows.append(f"{name:<{max_name}}  {cap:<{max_cap}}  {path:<{max_path}}  {stability}")

    return "\n".join([header, separator, *rows])


def list_examples(
    examples: List[Dict],
    capability: Optional[str] = None,
    tag: Optional[str] = None,
    stability: Optional[str] = None,
) -> str:
    """List examples with optional filtering (pure function).

    Args:
        examples: All examples from discovery
        capability: Optional capability filter
        tag: Optional tag filter
        stability: Optional stability filter

    Returns:
        Formatted table string
    """
    filtered = filter_examples(examples, capability, tag, stability)

    # Sort by capability, then name
    filtered.sort(key=lambda e: (e.get('capability', ''), e.get('name', '')))

    return format_table(filtered)
