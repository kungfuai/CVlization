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
    example_type: Optional[str] = None,
) -> List[Dict]:
    """Filter examples by criteria (pure function).

    Args:
        examples: List of example metadata dicts
        capability: Filter by capability (e.g., "perception", "generative")
        tag: Filter by tag (e.g., "ocr", "video")
        stability: Filter by stability (e.g., "stable", "beta")
        keyword: Filter by keyword in name, description, or tags
        example_type: Filter by type ("example", "benchmark", or None for all)

    Returns:
        Filtered list of examples
    """
    # Exclude archived examples by default
    filtered = [e for e in examples if not e.get('archived', False)]

    # Filter by type (example vs benchmark)
    if example_type:
        filtered = [e for e in filtered if e.get('_type', 'example') == example_type]

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


def format_type_badge(example_type: str) -> str:
    """Format a type badge with color.

    Args:
        example_type: Type string ('example' or 'benchmark')

    Returns:
        Colored badge string
    """
    # ANSI codes
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'

    if example_type == 'benchmark':
        return f"{MAGENTA}[benchmark]{RESET}"
    else:
        return f"{CYAN}[example]{RESET}"


def format_list(examples: List[Dict], keyword: Optional[str] = None, show_type: bool = True) -> str:
    """Format examples as a simple list.

    Args:
        examples: List of example metadata dicts
        keyword: Optional keyword to highlight in output
        show_type: Whether to show the type badge (default: True)

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
        example_type = e.get('_type', 'example')

        # Apply highlighting if keyword provided
        if keyword:
            name = highlight_matches(name, keyword)
            cap = highlight_matches(cap, keyword)
            path = highlight_matches(path, keyword)
            stability = highlight_matches(stability, keyword)

        # Format: name (capability) [type] - stability
        #           path
        if show_type:
            type_badge = format_type_badge(example_type)
            lines.append(f"{name} ({cap}) {type_badge} - {stability}")
        else:
            lines.append(f"{name} ({cap}) - {stability}")
        lines.append(f"  {path}")
        lines.append("")  # Blank line between entries

    # Remove trailing blank line
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def format_grouped(examples: List[Dict], keyword: Optional[str] = None) -> str:
    """Format examples as a grouped, compact view.

    Groups by top-level capability, then sub-category, showing example names
    inline with counts.

    Args:
        examples: List of example metadata dicts
        keyword: Optional keyword to highlight in output

    Returns:
        Formatted grouped view as string
    """
    if not examples:
        return "No examples found."

    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    # Group by primary capability -> sub-capability -> list of names
    # Only use first 2 levels: "analytical/tabular/causal" -> ("analytical", "tabular")
    groups: Dict[str, Dict[str, List[str]]] = {}
    for e in examples:
        cap = e.get('capability', '') or ''
        parts = cap.split('/')
        primary = parts[0] if parts[0] else 'uncategorized'
        sub = parts[1] if len(parts) > 1 and parts[1] else 'other'
        name = e.get('name', 'unknown')
        groups.setdefault(primary, {}).setdefault(sub, []).append(name)

    # Sort sub-categories by count descending within each group
    for primary in groups:
        for sub in groups[primary]:
            groups[primary][sub].sort()

    # Sort primary groups by count descending
    sorted_primaries = sorted(
        groups.items(),
        key=lambda item: sum(len(v) for v in item[1].values()),
        reverse=True,
    )

    lines = [f"Examples ({len(examples)} total)", ""]

    for primary, subs in sorted_primaries:
        primary_count = sum(len(names) for names in subs.values())
        primary_label = highlight_matches(primary, keyword) if keyword else primary
        lines.append(f"{BOLD}{primary_label}{RESET} ({primary_count})")

        sorted_subs = sorted(subs.items(), key=lambda item: len(item[1]), reverse=True)
        max_sub_len = max(len(sub) for sub, _ in sorted_subs)

        for sub, names in sorted_subs:
            # Truncate name list if too many
            MAX_INLINE = 3
            if keyword:
                display_names = [highlight_matches(n, keyword) for n in names]
            else:
                display_names = list(names)

            if len(display_names) > MAX_INLINE:
                shown = ", ".join(display_names[:MAX_INLINE])
                name_str = f"{shown}, {DIM}...{RESET} ({len(names)})"
            else:
                name_str = f"{', '.join(display_names)} ({len(names)})"

            sub_label = highlight_matches(sub, keyword) if keyword else sub
            lines.append(f"  {sub_label:<{max_sub_len}}  {name_str}")

        lines.append("")

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
    example_type: Optional[str] = None,
) -> str:
    """List examples with optional filtering (pure function).

    Args:
        examples: All examples from discovery
        capability: Optional capability filter
        tag: Optional tag filter
        stability: Optional stability filter
        keyword: Optional keyword filter
        format_type: Output format ('grouped', 'list', or 'table')
        example_type: Filter by type ('example', 'benchmark', or None for all)

    Returns:
        Formatted string
    """
    filtered = filter_examples(examples, capability, tag, stability, keyword, example_type)

    # Sort by type (benchmarks first), then capability, then name
    filtered.sort(key=lambda e: (
        0 if e.get('_type') == 'benchmark' else 1,
        e.get('capability', ''),
        e.get('name', '')
    ))

    # Choose formatter based on format_type
    if format_type == 'table':
        return format_table(filtered, keyword)
    elif format_type == 'list':
        return format_list(filtered, keyword)
    else:  # default to 'grouped'
        return format_grouped(filtered, keyword)
