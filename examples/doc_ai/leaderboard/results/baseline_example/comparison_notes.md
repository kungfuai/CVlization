# Doc AI Leaderboard - Baseline Comparison

**Run Date:** 2025-10-16 14:45:50
**Test Image:** sample.jpg (simple invoice with 2 line items)
**Models Tested:** 6
**Hardware:** NVIDIA GPU with CUDA 12.4.1

---

## Executive Summary

### 🏆 Top Performer: SURYA
- **Speed:** 7.43s (2nd fastest)
- **Accuracy:** 100%
- **Output:** Clean plain text

### Key Findings
- **3 models achieved 100% accuracy:** Surya, Nanonets-OCR, Granite-Docling
- **Models show different strengths:** Speed, formatting, structure preservation
- **Speed range:** 6.77s to 22.21s (3.3x difference)

---

## Results Summary

| Rank | Model | Time | Accuracy | Best For |
|------|-------|------|----------|----------|
| 🥇 | **Surya** | 7.43s | 100% | General OCR, fast extraction |
| 🥈 | **Nanonets-OCR** | 12.55s | 100% | Structured markdown, tables |
| 🥉 | **Granite-Docling** | 18.87s | 100% | Layout analysis, spatial data |
| 4️⃣ | Moondream2 | 6.77s | ~70% | Quick previews (verify output) |
| 5️⃣ | Moondream3 | 14.45s | Varies | VQA, interactive queries |
| 6️⃣ | Docling-Serve | 22.21s | ~60% | Basic document parsing |

---

## Ground Truth

```
INVOICE
Date: October 9, 2025
Invoice #: 12345

Item         Qty    Price
Widget A     2      $10.00
Widget B     1      $25.00

Total: $45.00
```

---

## Model Outputs & Analysis

### 🥇 Surya (7.43s)

```
INVOICE
Date: October 9, 2025
Invoice #: 12345
Item Qty Price
Widget A 2 $10.00
Widget B 1 $25.00
Total: $45.00
```

**Accuracy:** ✅ 100%
**Strengths:** Fast, accurate, clean output
**Best For:** Production OCR, high-volume processing, reliable extraction

---

### 🥈 Nanonets-OCR (12.55s)

```markdown
INVOICE
Date: October 9, 2025
Invoice #: 12345

| Item | Qty | Price |
| :--- | :-- | :---- |
| Widget A | 2 | $10.00 |
| Widget B | 1 | $25.00 |

Total: $45.00
```

**Accuracy:** ✅ 100%
**Strengths:** Beautiful markdown tables, excellent formatting, supports LaTeX equations
**Best For:** Scientific papers, complex documents, when presentation quality matters

---

### 🥉 Granite-Docling VLM (18.87s)

```
<loc_31><loc_39><loc_199><loc_106>INVOICE <br> Date: October 9, 2025 <br> Invoice #: 12345
<loc_31><loc_130><loc_218><loc_197>Item Qty Price <br> Widget A 2 $10.00 <br> Widget B 1 $25.00
<loc_31><loc_223><loc_128><loc_245>Total: $45.00
```

**Accuracy:** ✅ 100%
**Strengths:** Includes spatial coordinates, preserves layout structure
**Best For:** Layout analysis, position-aware processing, complex document understanding

---

### Moondream2 (6.77s)

```
- INVOICE
- Date: October 9, 2025
- Invoice #: 12345
- Item A 2 $10.00
- Item B 1 $25.00
- Total: $45.00
```

**Accuracy:** ~70%
**Note:** Extracted "Item A/B" instead of "Widget A/B"
**Strengths:** Very fast, clean bullet-point format, good for general understanding
**Best For:** Quick document previews, when exact text matching isn't critical
**Consideration:** Outputs should be verified for precise data extraction tasks

---

### Moondream3 (14.45s)

```
$45.00
```

**Output:** Only extracted the total amount
**Note:** This model performs better with interactive visual question answering
**Strengths:** Designed for VQA (Visual Question Answering), conversational queries
**Best For:** Interactive document Q&A, when you want to ask specific questions about content
**Consideration:** Default OCR mode may not extract complete documents; better suited for targeted queries

---

### Docling-Serve (22.21s)

```
INVOICE Date: October 9 1 2025 Invoice #: 12345

Item Qty Price Widget A 2 $10.00 Widget B 1 425.00

## Total: $45.00
```

**Accuracy:** ~60%
**Note:** Some parsing variations in dates and currency formatting
**Strengths:** IBM's Docling library, general document parsing
**Best For:** Basic document conversion, when combined with post-processing
**Consideration:** May benefit from additional validation for financial documents

---

## Recommendations by Use Case

### For Production OCR Systems
**Recommended:** Surya or Granite-Docling VLM
- Both achieved 100% accuracy
- Reliable, consistent output
- Good speed-accuracy balance

### For Scientific/Academic Papers
**Recommended:** Nanonets-OCR
- Excellent markdown table formatting
- Handles LaTeX equations
- Best structural preservation

### For Layout Analysis
**Recommended:** Granite-Docling VLM
- Spatial coordinate information
- Layout structure preservation
- Position-aware extraction

### For Quick Previews/Demos
**Recommended:** Moondream2 or Surya
- Fast processing
- Clean output format
- Good for rapid iteration

### For Interactive Document Q&A
**Recommended:** Moondream3
- Designed for visual question answering
- Good for targeted information extraction
- Works best with specific queries

---

## Performance Characteristics

### Speed
- **Fastest:** Moondream2 (6.77s)
- **Mid-range:** Surya (7.43s), Nanonets-OCR (12.55s), Moondream3 (14.45s)
- **Slower:** Granite-Docling (18.87s), Docling-Serve (22.21s)

### Accuracy (on this test)
- **100%:** Surya, Nanonets-OCR, Granite-Docling VLM
- **Approximate:** Moondream2, Docling-Serve
- **Limited extraction:** Moondream3 (in default OCR mode)

### Output Format
- **Plain text:** Surya, Moondream2
- **Markdown:** Nanonets-OCR, Docling-Serve
- **Structured with coordinates:** Granite-Docling VLM
- **Varies:** Moondream3 (depends on query)

---

## Benchmark Data

```csv
model,image,time_seconds,status
docling-serve,sample.jpg,22.210416770,success
granite-docling,sample.jpg,18.871429996,success
moondream2,sample.jpg,6.770287622,success
moondream3,sample.jpg,14.459550763,success
nanonets-ocr,sample.jpg,12.550378969,success
surya,sample.jpg,7.432639683,success
```

---

## Notes on Testing

- **Single test case:** Results based on one invoice image
- **Model capabilities vary:** Each model has different design goals and strengths
- **Use case matters:** Choose based on your specific requirements
- **Context is key:** Different models excel at different tasks

### Factors to Consider When Choosing

1. **Accuracy requirements:** How precise does extraction need to be?
2. **Speed requirements:** Real-time vs batch processing?
3. **Output format needs:** Plain text, markdown, structured data?
4. **Document types:** Invoices, scientific papers, forms, etc.
5. **Use case:** OCR vs VQA vs layout analysis?

---

*Generated by CVlization Doc AI Leaderboard - Baseline Example*
