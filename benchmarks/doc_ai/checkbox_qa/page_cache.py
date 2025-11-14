"""
Utilities for caching CheckboxQA PDF pages as PNG images.

Used by run_checkbox_qa.py and model adapters to lazily populate caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

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
    """
    Locate PDF files under data/documents.
    """
    docs: List[Path] = []
    if doc_ids:
        for doc_id in doc_ids:
            pdf = documents_dir / f"{doc_id}.pdf"
            if pdf.exists():
                docs.append(pdf)
    else:
        docs.extend(sorted(documents_dir.glob("*.pdf")))
    return docs
