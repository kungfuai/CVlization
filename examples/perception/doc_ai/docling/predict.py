#!/usr/bin/env python3
"""
Document extraction using Docling.

Supports PDF and image files, outputs structured JSON or Markdown.
Dual-mode execution: standalone or via CVL with --inputs/--outputs.
"""
import argparse
import json
from pathlib import Path
from docling.document_converter import DocumentConverter

# CVL dual-mode execution support
from cvlization.paths import (
    get_input_dir,
    get_output_dir,
    resolve_input_path,
    resolve_output_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract content from PDF or image files using Docling"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="sample.pdf",
        help="Path to input PDF or image file (default: sample.pdf)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/result.{format})"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "text"],
        default="json",
        help="Output format: json (default), markdown, or text"
    )
    parser.add_argument(
        "--export-tables",
        action="store_true",
        help="Export tables separately"
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="Export images metadata separately"
    )

    args = parser.parse_args()

    # Resolve paths for CVL dual-mode support
    INP = get_input_dir()
    OUT = get_output_dir()

    # Smart default for output path
    if args.output is None:
        # Use format-appropriate extension
        ext = {"json": "json", "markdown": "md", "text": "txt"}[args.format]
        args.output = f"result.{ext}"

    # Resolve paths using cvlization utilities
    input_path = Path(resolve_input_path(args.input, INP))
    output_path = Path(resolve_output_path(args.output, OUT))

    # Validate input file - if using default and not found, look in example directory
    if not input_path.exists() and args.input == "sample.pdf":
        # Try to find sample.pdf in the example directory
        script_dir = Path(__file__).parent
        example_sample = script_dir / "sample.pdf"
        if example_sample.exists():
            input_path = example_sample
            print(f"Note: Using sample file from example directory: {input_path}")
        else:
            print(f"Error: Input file '{input_path}' not found")
            print(f"\nTo run docling, either:")
            print(f"  1. Run from the docling directory: cd examples/perception/doc_ai/docling && cvl run docling predict")
            print(f"  2. Use -w flag: cvl run -w examples/perception/doc_ai/docling docling predict")
            print(f"  3. Specify your own PDF: cvl run docling predict --input your-file.pdf")
            return 1
    elif not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        return 1

    # Initialize converter
    print(f"Loading Docling models...", flush=True)
    converter = DocumentConverter()

    # Convert document
    print(f"\n{'='*80}")
    print("INPUT")
    print('='*80)
    print(f"Document: {input_path}")
    print('='*80 + '\n')

    print(f"Processing document...", flush=True)
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

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding='utf-8')

    # Show preview of output (first 500 chars)
    print("\n" + "="*80)
    print(f"{args.format.upper()} OUTPUT (preview):")
    print("="*80)
    preview = output_text[:500] + ("..." if len(output_text) > 500 else "")
    print(preview)
    print("="*80 + "\n")

    # Show container path (CVL will translate to host path)
    print(f"Output saved to {output_path}")

    # Export tables if requested
    if args.export_tables and hasattr(result.document, 'tables'):
        tables_dir = output_path.parent / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for i, table in enumerate(result.document.tables):
            table_file = tables_dir / f"table_{i:03d}.json"
            table_file.write_text(
                json.dumps(table.export_to_dict(), indent=2),
                encoding='utf-8'
            )
        print(f"Exported {len(result.document.tables)} tables to {tables_dir}")

    # Export images if requested
    if args.export_images and hasattr(result.document, 'pictures'):
        images_dir = output_path.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for i, picture in enumerate(result.document.pictures):
            # Save image metadata (actual image extraction depends on Docling version)
            image_file = images_dir / f"image_{i:03d}.json"
            image_file.write_text(json.dumps({
                "index": i,
                "caption": getattr(picture, 'caption', None),
                "bbox": getattr(picture, 'bbox', None),
            }, indent=2), encoding='utf-8')
        print(f"Exported {len(result.document.pictures)} images metadata to {images_dir}")

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
