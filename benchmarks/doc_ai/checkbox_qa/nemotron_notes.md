# Nemotron Parse v1.1 - CheckboxQA Evaluation Notes

## Summary

NVIDIA Nemotron Parse v1.1 is a document structure extraction model, **not suitable for checkbox QA tasks**. This document records our findings from testing it on CheckboxQA.

## Model Overview

- **Model**: nvidia/NVIDIA-Nemotron-Parse-v1.1
- **Size**: <1B parameters
- **Architecture**: ViT-H encoder (C-RADIO) + mBart decoder
- **Task**: Document structure extraction with spatial grounding
- **Output**: Markdown + bounding boxes + semantic class labels

## Key Limitations for CheckboxQA

### 1. Cannot Detect Checked Checkbox State

| Visual Element | Nemotron Output | Detection |
|----------------|-----------------|-----------|
| ■ (filled/checked box) | Nothing - completely ignored | ❌ |
| □ (empty/unchecked box) | `\square` (LaTeX) | ✓ (sometimes) |
| ✓ / X (checkmarks) | Nothing | ❌ |

**Example from doc 016f73a5:**
- Visual: `■ Original  □ Amendment`
- Output: `_Original_ \(\square\) **Amendment**`
- The checked box (■) was invisible; only the empty box (□) was detected.

### 2. No Checkbox Semantic Class

Available classes: `Bibliography, Caption, Code, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, TOC, Table, Text, Title`

**No `Checkbox`, `Form-field`, or `Input` class exists.**

### 3. Single Image Input Only

```python
limit_mm_per_prompt={"image": 1}  # vLLM config
```

Cannot process multi-page documents in a single inference call (unlike Qwen3-VL which handles 5+ pages).

### 4. No Constrained Generation / Structured Output

- Fixed output format: `<x_><y_>text<x_><y_><class_>`
- No Pydantic/JSON schema support
- Cannot force VQA-style answers

## Test Results

### Test 1: Document with Unchecked Box (016f73a5)

**Header checkboxes**: "■ Original □ Amendment"

```
Output: _Original_ \(\square\) **Amendment**
```

- ✓ Detected empty box as `\square`
- ❌ Did NOT detect filled box (■) - just output "Original" without indicator

### Test 2: Document with Checked Boxes (2c5bfa6b - ATF Form 4473)

**Row 18 "Type of firearm"**: Handgun has X mark (checked)

```
Output: Handgun & Long Gun & Other Firearm (Frame, Receiver, etc.
```

- ❌ No checkbox symbols detected at all
- ❌ Cannot determine which option is selected

## Why This Happens

1. **Training data**: Academic/technical documents, not government forms
2. **Checkbox as noise**: Small marks (X, ✓) treated as noise, not content
3. **Layout focus**: Model extracts document structure, not form field states
4. **OCR limitation**: Not trained to recognize handwritten marks in boxes

## Potential Workarounds (Not Recommended)

### Approach 1: Absence-based Detection
- If `\square` present → unchecked
- If `\square` absent → possibly checked

**Problems**: Requires knowing which fields should have checkboxes; unreliable.

### Approach 2: Two-Stage Pipeline
1. Nemotron Parse → extract markdown
2. LLM (Claude/GPT) → answer questions from markdown

**Problems**: 2x latency, higher cost, markdown may be incomplete.

### Approach 3: Fine-tuning
Train Nemotron on form documents with checkbox annotations.

**Problems**: Requires training data, compute, and may break other capabilities.

## Actual Benchmark Results

**Nemotron Parse on CheckboxQA dev subset (24 questions):**

| Model | ANLS* | Notes |
|-------|-------|-------|
| Qwen3-VL-4B | 0.44 | Best VQA model |
| Phi-4 | 0.38 | Good VQA model |
| Qwen3-VL-2B | 0.35 | Smaller VQA model |
| **Nemotron Parse** | **0.11** | Document parsing (not suitable) |

The 0.11 ANLS* score is essentially random guessing - confirms Nemotron Parse cannot detect checkbox states.

## Recommendation

**Use VQA models (Qwen3-VL, Phi-4) for CheckboxQA** instead of document parsing models.

| Model | Task Type | CheckboxQA Suitable |
|-------|-----------|---------------------|
| Qwen3-VL-4B | VQA | ✓ Best (0.38 ANLS* test) |
| Phi-4 | VQA | ✓ Good (0.37 ANLS* test) |
| Nemotron Parse | Doc Parsing | ❌ Not suitable (0.11 ANLS*) |

## Test Script

See `test_nemotron_parse.py` for reproducing these findings:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v ~/.cache/cvlization:/root/.cache/cvlization \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  cvlization/nvidia-nemotron-parse:latest \
  python test_nemotron_parse.py --doc-id 016f73a5 --page 1
```

## References

- Model: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
- Paper: arXiv:2511.20478
- CheckboxQA: https://huggingface.co/datasets/mturski/CheckboxQA
