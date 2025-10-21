#!/usr/bin/env python3
"""
Document extraction using Docling.
Supports PDF and image files, outputs structured JSON or Markdown.
"""
import argparse
import json
from pathlib import Path
from docling.document_converter import DocumentConverter

def main():
    parser = argparse.ArgumentParser(description="Extract content from PDF or image files using Docling")
    parser.add_argument("input_file", type=str, help="Path to input PDF or image file")
    parser.add_argument("--output", type=str, help="Output file path (optional, prints to stdout if not specified)")
    parser.add_argument("--format", type=str, choices=["json", "markdown", "text"], default="json",
                       help="Output format: json (default), markdown, or text")
    parser.add_argument("--export-tables", action="store_true", help="Export tables separately")
    parser.add_argument("--export-images", action="store_true", help="Export images separately")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1

    # Initialize converter
    print(f"Loading Docling models...", flush=True)
    converter = DocumentConverter()

    # Convert document
    print(f"Processing: {input_path.name}", flush=True)
    result = converter.convert(str(input_path))

    # Format output
    if args.format == "json":
        # Export to JSON with full structure
        output_data = {
            "input_file": str(input_path),
            "pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1,
            "content": result.document.export_to_dict(),
        }
        output_text = json.dumps(output_data, indent=2, ensure_ascii=False)
    elif args.format == "markdown":
        output_text = result.document.export_to_markdown()
    else:  # text
        output_text = result.document.export_to_text()

    # Write or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding='utf-8')
        print(f"Output written to: {output_path}")
    else:
        print("\n" + "="*80)
        print(output_text)
        print("="*80)

    # Export tables if requested
    if args.export_tables and hasattr(result.document, 'tables'):
        tables_dir = Path(args.output).parent / "tables" if args.output else Path("./outputs/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        for i, table in enumerate(result.document.tables):
            table_file = tables_dir / f"table_{i:03d}.json"
            table_file.write_text(json.dumps(table.export_to_dict(), indent=2), encoding='utf-8')
        print(f"Exported {len(result.document.tables)} tables to: {tables_dir}")

    # Export images if requested
    if args.export_images and hasattr(result.document, 'pictures'):
        images_dir = Path(args.output).parent / "images" if args.output else Path("./outputs/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        for i, picture in enumerate(result.document.pictures):
            # Save image metadata (actual image extraction depends on Docling version)
            image_file = images_dir / f"image_{i:03d}.json"
            image_file.write_text(json.dumps({
                "index": i,
                "caption": getattr(picture, 'caption', None),
                "bbox": getattr(picture, 'bbox', None),
            }, indent=2), encoding='utf-8')
        print(f"Exported {len(result.document.pictures)} images metadata to: {images_dir}")

    return 0

if __name__ == "__main__":
    exit(main())
