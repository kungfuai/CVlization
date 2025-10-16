# Extract-0: Structured Document Extraction (Placeholder)

> **Status**: 🚧 Placeholder for future implementation
>
> This directory is reserved for Extract-0, a specialized language model for structured document information extraction. Implementation pending availability of pre-trained model weights.

## Overview

**Extract-0** is a 7-billion parameter language model specifically optimized for extracting structured JSON data from document text according to user-defined schemas. It achieves state-of-the-art performance on extraction tasks, outperforming models with orders of magnitude more parameters.

### Key Characteristics

- **Model Type**: Text-to-structured-data extraction (post-OCR)
- **Base Model**: DeepSeek-R1-Distill-Qwen-7B with LoRA adapters
- **Input**: Plain text (from OCR or documents)
- **Output**: Structured JSON matching provided schema
- **Performance**: Mean reward 0.573 vs GPT-4.1 (0.457), o3 (0.464)
- **JSON Validity**: 89% in production scenarios

## How It Differs from Other Examples

| Category | Examples | Purpose | Input | Output |
|----------|----------|---------|-------|--------|
| **OCR/VLM** | nanonets-ocr, granite-docling, moondream, surya | Document → Text | Images/PDFs | Markdown/Text |
| **Structured Extraction** | **extract0** | Text → Structured Data | Text | JSON |

Extract-0 is a **post-processing step** that complements OCR models:

```
PDF/Image → [OCR Model] → Text → [Extract-0] → Structured JSON
```

## Use Cases

- **Form Processing**: Extract structured fields from registration forms, applications
- **Invoice Extraction**: Pull amounts, dates, line items into structured format
- **Scientific Papers**: Extract authors, dates, findings, citations
- **Medical Records**: Structure patient data, diagnoses, prescriptions
- **Regulatory Documents**: Extract compliance data, requirements, dates

## Example Workflow

```bash
# Step 1: OCR with nanonets-ocr
cd ../nanonets-ocr
bash predict.sh invoice.pdf --output invoice_text.md

# Step 2: Extract structured data with extract0 (when implemented)
cd ../extract0
bash predict.sh invoice_text.md --schema invoice_schema.json --output invoice_data.json
```

Example schema:
```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string"},
    "date": {"type": "string", "format": "date"},
    "total_amount": {"type": "number"},
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "quantity": {"type": "number"},
          "price": {"type": "number"}
        }
      }
    }
  }
}
```

## Why Not Implemented Yet?

### Current Blockers

1. **No Pre-trained Weights Available**
   - GitHub repo contains only training code
   - No model weights published on HuggingFace
   - Would require ~8 hours of training on H100 GPU ($196 cost)

2. **Training Required**
   - Users would need to:
     - Generate 280k synthetic training examples
     - Run supervised fine-tuning (SFT)
     - Run reinforcement learning with GRPO
   - Not aligned with "quick start" examples in CVlization

3. **Dependency on Other Models**
   - Requires OCR output as input
   - Not a self-contained example like others

### What Would Be Needed

To implement this example, we would need:

- [ ] Pre-trained model weights published on HuggingFace
- [ ] Simple inference-only API (without training pipeline)
- [ ] Example integration with nanonets-ocr or granite-docling
- [ ] Dockerfile with all dependencies
- [ ] Sample schemas and test documents

## Resources

### Official Links

- **GitHub Repository**: https://github.com/herniqeu/extract0
- **arXiv Paper**: https://arxiv.org/abs/2509.22906
- **Dataset**: https://huggingface.co/datasets/HenriqueGodoy/extract-0
- **Base Model**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

### Technical Details

- **Training Dataset**: 280,128 synthetic extraction examples from arXiv, PubMed, Wikipedia, FDA docs
- **Architecture**: LoRA rank 16, alpha 32
- **Trainable Parameters**: 40.4M (0.53% of total 7.66B)
- **Training Method**: Supervised fine-tuning + GRPO (Group Relative Policy Optimization)
- **Inference Requirements**: 24GB VRAM recommended

### Key Paper Insights

- Semantic similarity reward function for handling extraction ambiguity
- Field-level matching using sentence embeddings (MiniLM-L6-v2)
- Memory-preserving chunk processing (2000 chars with 200 char overlap)
- 35.4% reward improvement from reinforcement learning stage

## How to Revisit This

### Checklist for Future Implementation

1. **Check for Pre-trained Model**
   ```bash
   # Search HuggingFace for published weights
   # Look for: HenriqueGodoy/extract0 or herniqeu/extract0
   ```

2. **Test Inference**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import PeftModel

   # If weights become available:
   base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
   model = PeftModel.from_pretrained(base, "HenriqueGodoy/extract0")
   ```

3. **Create Dockerized Example**
   - Follow pattern from nanonets-ocr
   - Add schema-based extraction API
   - Include example pipelines with other OCR models

4. **Integration Points**
   - Add as optional step in nanonets-ocr README
   - Create combined pipeline example
   - Document schema design best practices

## Alternative Solutions (Currently Available)

While waiting for Extract-0 implementation, consider:

1. **VLM-based Extraction**: Use nanonets-ocr or granite-docling in VQA mode
   ```bash
   bash predict.sh invoice.pdf --mode vqa --question "Extract invoice number, date, and total"
   ```

2. **Prompt Engineering**: Use moondream3 with structured prompts
   ```bash
   bash predict.sh form.pdf --mode vqa --question "Return JSON with fields: name, date, amount"
   ```

3. **Post-processing with GPT-4**: Pipe OCR output through OpenAI API
   ```python
   # After OCR
   response = openai.chat.completions.create(
       model="gpt-4",
       messages=[{
           "role": "user",
           "content": f"Extract this schema from text: {schema}\n\nText: {ocr_text}"
       }]
   )
   ```

## Citation

```bibtex
@misc{godoy2025extract0specializedlanguagemodel,
      title={Extract-0: A Specialized Language Model for Document Information Extraction},
      author={Henrique Godoy},
      year={2025},
      eprint={2509.22906},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22906},
}
```

## Contact for Updates

If you find pre-trained weights or want to implement this:

1. Check GitHub issues: https://github.com/herniqeu/extract0/issues
2. Check HuggingFace model hub: https://huggingface.co/models?search=extract0
3. Update this README and implement following the nanonets-ocr pattern
4. Notify maintainers that implementation is ready

---

**Last Updated**: 2025-10-15
**Status**: Monitoring for pre-trained model release
**Priority**: Medium - Would be valuable complement to existing OCR examples
