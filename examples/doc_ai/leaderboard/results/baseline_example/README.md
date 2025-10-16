# Baseline Example Results

This directory contains a complete reference benchmark run showing all 6 models tested on the same invoice image.

## Quick Summary

- **Test Image:** `../../test_data/sample.jpg`
- **Models:** 6 (all GPU-accelerated)
- **Top performer:** Surya (7.43s, 100% accurate)
- **Date:** 2025-10-16

## Files

- `benchmark.csv` - Raw timing data
- `comparison_notes.md` - Detailed analysis with recommendations
- `run_info.txt` - Run metadata
- `*/sample_output.txt` - Individual model outputs
- `*/sample_log.txt` - Execution logs

## Results Overview

**Achieved 100% Accuracy:**
- Surya (7.43s) - General OCR
- Nanonets-OCR (12.55s) - Markdown tables
- Granite-Docling VLM (18.87s) - Layout analysis

**Other Models:**
- Moondream2 (6.77s) - Fast previews
- Moondream3 (14.45s) - VQA/interactive queries
- Docling-Serve (22.21s) - Document parsing

Each model has different strengths and is suited for different use cases. See `comparison_notes.md` for detailed analysis and recommendations.
