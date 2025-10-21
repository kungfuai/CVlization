#!/usr/bin/env python3
"""Convert first page of PDF to image for testing."""
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("sample.pdf")
page = pdf[0]
bitmap = page.render(scale=2.0)
pil_image = bitmap.to_pil()
pil_image.save("test_image.png")
print("Created test_image.png from first page of sample.pdf")
