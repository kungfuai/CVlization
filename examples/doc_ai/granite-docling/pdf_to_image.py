#!/usr/bin/env python3
"""Convert PDF to images for Granite-Docling processing."""
import sys
from pathlib import Path
import pypdfium2 as pdfium

if len(sys.argv) < 2:
    print("Usage: python pdf_to_image.py <input.pdf> [output_prefix]")
    sys.exit(1)

pdf_path = Path(sys.argv[1])
output_prefix = sys.argv[2] if len(sys.argv) > 2 else pdf_path.stem

if not pdf_path.exists():
    print(f"Error: PDF file '{pdf_path}' not found")
    sys.exit(1)

pdf = pdfium.PdfDocument(pdf_path)
print(f"Converting {len(pdf)} pages from {pdf_path.name}...")

for i, page in enumerate(pdf):
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    output_file = f"{output_prefix}_page{i+1:03d}.png"
    pil_image.save(output_file)
    print(f"  Created: {output_file}")

print(f"Done! Converted {len(pdf)} pages.")
