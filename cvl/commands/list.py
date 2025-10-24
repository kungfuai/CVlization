"""List command - display available examples."""
from typing import List, Dict, Optional
import re


def highlight_matches(text: str, keyword: Optional[str]) -> str:
    """Highlight keyword matches in text using ANSI codes.

    Args:
        text: The text to highlight
        keyword: The keyword to highlight (case-insensitive)

    Returns:
        Text with highlighted matches
    """
    if not keyword or not text:
        return text

    # ANSI codes: bold + yellow
    HIGHLIGHT = '\033[1;33m'
    RESET = '\033[0m'

    # Case-insensitive replacement
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f'{HIGHLIGHT}{m.group()}{RESET}', text)


def filter_examples(
    examples: List[Dict],
    capability: Optional[str] = None,
    tag: Optional[str] = None,
    stability: Optional[str] = None,
    keyword: Optional[str] = None,
) -> List[Dict]:
    """Filter examples by criteria (pure function).

    Args:
        examples: List of example metadata dicts
        capability: Filter by capability (e.g., "perception", "generative")
        tag: Filter by tag (e.g., "ocr", "video")
        stability: Filter by stability (e.g., "stable", "beta")
        keyword: Filter by keyword in name, description, or tags

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

    if keyword:
        kw_lower = keyword.lower()
        filtered = [
            e for e in filtered
            if (kw_lower in e.get('name', '').lower() or
                kw_lower in e.get('description', '').lower() or
                any(kw_lower in t.lower() for t in e.get('tags', [])))
        ]

    return filtered


def format_table(examples: List[Dict], keyword: Optional[str] = None) -> str:
    """Format examples as a simple table.

    Args:
        examples: List of example metadata dicts
        keyword: Optional keyword to highlight in output

    Returns:
        Formatted table as string
    """
    if not examples:
        return "No examples found."

    # Calculate column widths (using plain text for width calculation)
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

        # Apply highlighting if keyword provided
        if keyword:
            name = highlight_matches(name, keyword)
            cap = highlight_matches(cap, keyword)
            path = highlight_matches(path, keyword)
            stability = highlight_matches(stability, keyword)

        rows.append(f"{name:<{max_name}}  {cap:<{max_cap}}  {path:<{max_path}}  {stability}")

    return "\n".join([header, separator, *rows])


def format_list(examples: List[Dict], keyword: Optional[str] = None) -> str:
    """Format examples as a simple list.

    Args:
        examples: List of example metadata dicts
        keyword: Optional keyword to highlight in output

    Returns:
        Formatted list as string
    """
    if not examples:
        return "No examples found."

    lines = []
    for e in examples:
        name = e.get('name', 'unknown')
        cap = e.get('capability', '')
        path = e.get('_path', '')
        stability = e.get('stability', 'unknown')

        # Apply highlighting if keyword provided
        if keyword:
            name = highlight_matches(name, keyword)
            cap = highlight_matches(cap, keyword)
            path = highlight_matches(path, keyword)
            stability = highlight_matches(stability, keyword)

        # Format: name (capability) - stability
        #           path
        lines.append(f"{name} ({cap}) - {stability}")
        lines.append(f"  {path}")
        lines.append("")  # Blank line between entries

    # Remove trailing blank line
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def list_examples(
    examples: List[Dict],
    capability: Optional[str] = None,
    tag: Optional[str] = None,
    stability: Optional[str] = None,
    keyword: Optional[str] = None,
    format_type: str = 'list',
) -> str:
    """List examples with optional filtering (pure function).

    Args:
        examples: All examples from discovery
        capability: Optional capability filter
        tag: Optional tag filter
        stability: Optional stability filter
        keyword: Optional keyword filter
        format_type: Output format ('list' or 'table')

    Returns:
        Formatted string
    """
    filtered = filter_examples(examples, capability, tag, stability, keyword)

    # Sort by capability, then name
    filtered.sort(key=lambda e: (e.get('capability', ''), e.get('name', '')))

    # Choose formatter based on format_type
    if format_type == 'table':
        return format_table(filtered, keyword)
    else:  # default to 'list'
        return format_list(filtered, keyword)
