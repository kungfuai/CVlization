"""Utilities for caching CheckboxQA PDF pages as PNG images.

Used by run_checkbox_qa.py and model adapters to lazily populate caches.

This module now also exposes a tiny CLI:

    python page_cache.py --pdf path/to/doc.pdf --doc-id DOC123 --cache-dir data/page_images

to mirror the convenience that tests expect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import pypdfium2 as pdfium


DEFAULT_SCALE = 2.5


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    scale: float = DEFAULT_SCALE,
) -> int:
    """
    Render every page of a PDF to PNG files.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Target directory for page PNGs.
        overwrite: Re-render pages even if PNG already exists.
        scale: Rendering scale factor for pdfium (affects resolution).

    Returns:
        Number of pages rendered.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pdfium.PdfDocument(pdf_path)
    for index, page in enumerate(doc, start=1):
        target = output_dir / f"page-{index:03d}.png"
        if target.exists() and not overwrite:
            continue
        pil_image = page.render(scale=scale).to_pil()
        pil_image.save(target)

    return len(doc)


def list_cached_pages(cache_dir: Path) -> List[Path]:
    """Return sorted list of cached page PNGs for a document."""
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("page-*.png"))


# Default cache locations
_LOCAL_PAGE_CACHE = Path(__file__).parent / "data/page_images"
_SYSTEM_PAGE_CACHE = Path.home() / ".cache/cvlization/data/checkbox_qa/page_images"
_SYSTEM_DOCS_DIR = Path.home() / ".cache/cvlization/data/checkbox_qa/documents"


def get_page_cache_root() -> Path:
    """Get the page cache root directory, preferring system cache."""
    if _SYSTEM_PAGE_CACHE.exists():
        return _SYSTEM_PAGE_CACHE
    return _LOCAL_PAGE_CACHE


def ensure_page_cache(doc_id: str, page_cache_root: Optional[Path] = None) -> bool:
    """
    Ensure page images are rendered for a document. Returns True if successful.

    Lazily renders PDF pages to PNG if not already cached.
    """
    import sys

    if page_cache_root is None:
        page_cache_root = get_page_cache_root()

    doc_cache = page_cache_root / doc_id

    # Already cached?
    if doc_cache.exists() and list(doc_cache.glob("page-*.png")):
        return True

    # Find PDF
    pdf_path = _SYSTEM_DOCS_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        # Try local data/documents
        local_pdf = Path(__file__).parent / "data/documents" / f"{doc_id}.pdf"
        if local_pdf.exists():
            pdf_path = local_pdf
        else:
            print(f"Warning: PDF not found for {doc_id}", file=sys.stderr)
            return False

    # Render pages
    try:
        print(f"Rendering page images for {doc_id}...", file=sys.stderr)
        doc_cache.mkdir(parents=True, exist_ok=True)
        render_pdf_to_images(pdf_path, doc_cache)
        return True
    except Exception as e:
        print(f"Error rendering {doc_id}: {e}", file=sys.stderr)
        return False


def get_page_images(doc_id: str, max_pages: int = 20, page_cache_root: Optional[Path] = None) -> List[Path]:
    """
    Get list of page image paths for a document. Renders lazily if needed.

    Args:
        doc_id: Document ID (e.g., "2ba32ee0")
        max_pages: Maximum number of pages to return (None for all)
        page_cache_root: Override the page cache directory

    Returns:
        List of Path objects to page PNG files
    """
    if page_cache_root is None:
        page_cache_root = get_page_cache_root()

    doc_cache = page_cache_root / doc_id

    # Try lazy rendering if not cached
    if not doc_cache.exists() or not list(doc_cache.glob("page-*.png")):
        if not ensure_page_cache(doc_id, page_cache_root):
            return []

    page_files = sorted(doc_cache.glob("page-*.png"))
    if max_pages and len(page_files) > max_pages:
        page_files = page_files[:max_pages]

    return page_files


def load_document_ids_from_subset(subset_path: Path) -> List[str]:
    """
    Read a CheckboxQA subset jsonl file and return document IDs.
    """
    ids: List[str] = []
    with open(subset_path, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(item["name"])
    return ids


def resolve_pdf_paths(
    documents_dir: Path,
    doc_ids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Locate PDF files under data/documents."""
    docs: List[Path] = []
    if doc_ids:
        for doc_id in doc_ids:
            pdf = documents_dir / f"{doc_id}.pdf"
            if pdf.exists():
                docs.append(pdf)
    else:
        docs.extend(sorted(documents_dir.glob("*.pdf")))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render CheckboxQA PDFs into cached PNG pages")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF to render")
    parser.add_argument("--doc-id", required=True, help="Document ID (used for directory naming)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/page_images"), help="Root cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render even if cached")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Render scale for pdfium")

    args = parser.parse_args()

    target_dir = args.cache_dir / args.doc_id
    count = render_pdf_to_images(args.pdf, target_dir, overwrite=args.overwrite, scale=args.scale)
    print(f"Rendered {count} pages to {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
