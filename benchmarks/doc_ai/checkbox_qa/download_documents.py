#!/usr/bin/env python3
"""
CheckboxQA Document Downloader

Downloads PDFs from CheckboxQA dataset and extracts specified pages.
Matches the official CheckboxQA dowload_documents.py logic.
"""

import argparse
import json
from pathlib import Path
from logging import warning

import requests
from tqdm import tqdm

try:
    from PyPDF2 import PdfWriter, PdfReader
except ImportError:
    print("Error: PyPDF2 not installed. Install with: pip install PyPDF2")
    exit(1)


def extract_pages(pdf_path: Path, pages: list):
    """
    Extract specific pages from a PDF and overwrite the original file.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers (1-indexed) to extract
    """
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for page_num in pages:
        writer.add_page(reader.pages[page_num - 1])  # Page numbers are 1-based in the JSON

    with open(pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)


def download_pdfs(json_path: Path, out_dir: Path):
    """
    Download PDFs from URLs in document_url_map.json

    Matches official CheckboxQA logic:
    - Downloads PDF from pdf_url
    - If 'pages' is specified and not null, extracts only those pages
    - If 'pages' is null, keeps full PDF

    Args:
        json_path: Path to document_url_map.json
        out_dir: Directory to save PDFs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Iterate over the entries and download the PDFs
    for key, value in tqdm(data.items(), desc="Downloading PDFs"):
        pdf_url = value['pdf_url']
        pdf_response = requests.get(pdf_url)

        if pdf_response.status_code == 200:
            pdf_path = out_dir / f"{key}.pdf"

            # Save PDF
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)

            # Extract specific pages if specified
            if 'pages' in value and value['pages']:
                print(f"  Extracting pages {value['pages']} from {key}.pdf")
                extract_pages(pdf_path, value['pages'])
        else:
            warning(f"Failed to download {pdf_url} (status: {pdf_response.status_code})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download PDF files for CheckboxQA dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all PDFs using default paths
  python download_documents.py

  # Specify custom paths
  python download_documents.py --json_path data/document_url_map.json --out_dir data/documents

Notes:
  - If 'pages' field is null, the full PDF is kept
  - If 'pages' field contains page numbers, only those pages are extracted
  - Page numbers in JSON are 1-indexed
"""
    )

    parser.add_argument(
        '--out_dir',
        type=Path,
        default='./data/documents',
        help='Where to store the downloaded PDFs (default: ./data/documents)'
    )
    parser.add_argument(
        '--json_path',
        type=Path,
        default='./data/document_url_map.json',
        help='Path to the JSON file containing the document URL map (default: ./data/document_url_map.json)'
    )

    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: JSON file not found at {args.json_path}")
        exit(1)

    print("=" * 60)
    print("CheckboxQA Document Downloader")
    print("=" * 60)
    print(f"JSON path: {args.json_path}")
    print(f"Output directory: {args.out_dir}")
    print()

    download_pdfs(args.json_path, args.out_dir)

    print()
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
