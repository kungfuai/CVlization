# OMR Model Comparison — Biddle's Piano Waltz (1884)

Zero-shot OMR outputs from all models on the same vintage score:
**Biddle's Piano Waltz** (Robert D. Biddle, 1884), Library of Congress.
Image: `zzsi/cvl` → `qwen3_omr/vintage_score_1884.jpg`

Ground truth (visual inspection):
- **Time signature**: 3/4
- **Key signature**: ~2 sharps (D major) in intro; key change later in piece
- **Era**: Romantic / late 19th century (copyright 1884)

## Generate Report

```bash
python generate_report.py          # outputs report.html
python generate_report.py --output my_report.html
```

Opens in any browser. Score image is embedded (self-contained). No dependencies beyond the standard library.

## Results

| File | Model | Type |
|------|-------|------|
| `gemini_omr.json` | gemini/gemini-2.5-flash | API |
| `gemini3_omr.json` | gemini/gemini-3.1-pro-preview | API |
| `gpt4o_omr.json` | gpt-4o | API |
| `gpt52_omr.json` | gpt-5.2 | API |
| `gpt52pro_omr.json` | gpt-5.2-pro | API |
| `claude_sonnet_omr.json` | anthropic/claude-sonnet-4-6 | API |
| `qwen3_omr.json` | Qwen/Qwen3-VL-8B-Instruct | Local GPU |
| `smt_omr.json` | PRAIG/smt-grandstaff | Local GPU (fine-tuned) |

## Summary

| Model | Key Sig | Time Sig | Era | Dynamics | ekern |
|-------|---------|----------|-----|----------|-------|
| Gemini 2.5 Flash | G major (1♯) ❌ | 3/4 ✅ | ✅ | Good | Aborted ❌ |
| Gemini 3.1 Pro ⭐ | 1♯ + key change noticed | 3/4 ✅ | ✅ | Good | Best VLM attempt ✅ |
| GPT-4o | D major (2♯) ✅? | 3/4 ✅ | ✅ | Good | Refused ❌ |
| GPT-5.2 | C major (0) ❌ | 3/4 ✅ | ✅ | Most detailed | Declined gracefully |
| GPT-5.2 Pro | Bb major (2♭) ❌ | 3/4 ✅ | ✅ | Very detailed | Hit max_tokens |
| Claude Sonnet 4.6 | Bb major (2♭) ❌ | 3/4 ✅ | ✅ | Most structured | Partial notes ✅ |
| Qwen3-VL-8B (local) | F# major (1♯) ❌ | 3/4 ✅ | ✅ | Good | Header loop ❌ |
| SMT-OMR (fine-tuned) | F major (1♭) ❌? | **2/4 ❌** | N/A | N/A | Complete bekern ✅ |

**Run date**: 2026-02-23
