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
