# Shared Test Images for VLM Examples

This directory contains test images shared across all Vision Language Model examples to avoid duplication in git.

## Images

- **sample.jpg** (22KB) - General purpose test image for VLM inference

## Usage

All VLM examples in `examples/perception/vision_language/` can reference these images using relative paths:

```bash
# From any VLM example directory:
python predict.py --image ../test_images/sample.jpg
```

## Adding New Test Images

When adding new shared test images:
1. Keep file sizes small (<100KB when possible)
2. Use descriptive names
3. Add to this README
4. Ensure images are appropriate for OCR, captioning, and VQA tasks
