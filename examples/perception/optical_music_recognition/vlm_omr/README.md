# vlm-omr — Frontier VLM Zero-Shot OMR Probe

Provider-agnostic zero-shot Optical Music Recognition using frontier Vision-Language Models via [LiteLLM](https://docs.litellm.ai/). Routes to **Gemini 2.5 Flash** (default), GPT-4o, or Claude Opus/Sonnet without code changes — just set the appropriate API key.

Sends the same 5 music-theory prompts as the sibling [`qwen3-omr`](../qwen3_vl/) example, enabling direct comparison of API-based frontier models against a locally-hosted open-source VLM. Also supports **batch mode** for processing a directory of vintage scans and producing a JSONL metadata label file — useful for bootstrapping weak ground truth at scale (~$11–50 per 10k pages with Gemini 2.5 Flash).

No GPU required.

## Sample

The default sample is **Biddle's Piano Waltz** (Robert D. Biddle, 1884), from the [Library of Congress American Sheet Music collection](https://www.loc.gov/item/2023852770/). Public domain. Same image as `qwen3-omr` for direct comparison.

![Vintage score](https://huggingface.co/datasets/zzsi/cvl/resolve/main/qwen3_omr/vintage_score_1884.jpg)

## Quick Start

```bash
export GEMINI_API_KEY=your-key-here
./build.sh
./predict.sh                                         # Gemini 2.5 Flash, default sample
./predict.sh --image my_score.jpg                    # custom image
./predict.sh --model gpt-4o                          # GPT-4o
./predict.sh --model anthropic/claude-opus-4-6       # Claude Opus 4
./predict.sh --model anthropic/claude-sonnet-4-6     # Claude Sonnet 4
./predict.sh --prompts key_signature time_signature  # run subset only
```

## Batch Mode

Process a directory of vintage scans and produce a JSONL label file:

```bash
# Process all images in ./scans/, output one JSON record per line
./predict.sh --input-dir ./scans/ --output omr_results.jsonl

# Resume an interrupted run (skips files already in the JSONL)
./predict.sh --input-dir ./scans/ --output omr_results.jsonl --resume
```

Each line of the JSONL is one self-contained JSON record. The `--resume` flag reads existing output lines and skips already-processed filenames — safe to restart after interruption.

## Supported Models

| Model string | Provider | Key env var |
|---|---|---|
| `gemini/gemini-2.5-flash` (default) | Google | `GEMINI_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `anthropic/claude-opus-4-6` | Anthropic | `ANTHROPIC_API_KEY` |
| `anthropic/claude-sonnet-4-6` | Anthropic | `ANTHROPIC_API_KEY` |

Any other [LiteLLM-supported model](https://docs.litellm.ai/docs/providers) can be passed via `--model`.

## Prompts

| ID | Label | Expected quality |
|----|-------|-----------------|
| `key_signature` | Key signature | Good — visually salient |
| `time_signature` | Time signature | Good — visually salient |
| `musical_era` | Musical era | Moderate |
| `dynamics_tempo` | Dynamics/tempo markings | Moderate |
| `ekern_transcription` | Full ekern transcription | Poor — lower-bound probe |

## Output Format

### Single image (`omr_results.json`)

```json
{
  "file": "/path/to/vintage_score_1884.jpg",
  "model": "gemini/gemini-2.5-flash",
  "timestamp": "2026-02-23T14:00:00+00:00",
  "results": {
    "key_signature": {
      "label": "Key signature",
      "expectation": "Expected to work well ...",
      "prompt": "Look at this sheet music image ...",
      "response": "The key signature shows 3 flats (Bb, Eb, Ab), indicating Eb major."
    },
    "time_signature": { "...": "..." },
    "musical_era": { "...": "..." },
    "dynamics_tempo": { "...": "..." },
    "ekern_transcription": { "...": "..." }
  }
}
```

### Batch mode (`omr_results.jsonl`)

One JSON record per line. The `prompt` and `expectation` fields are omitted per line (static, reconstructible from source) to keep the file compact.

```jsonl
{"file": "scan_001.jpg", "model": "gemini/gemini-2.5-flash", "timestamp": "...", "results": {"key_signature": {"label": "Key signature", "response": "..."}, ...}}
{"file": "scan_002.jpg", ...}
```

## Comparison with sibling examples

| | `vlm-omr` (this) | `qwen3-omr` | `smt-omr` |
|--|---|---|---|
| Model | Frontier VLM API | Qwen3-VL-8B (local) | SMT transformer (local) |
| GPU required | No | Yes (~8 GB) | Yes (~4 GB) |
| Cost | API fees (~$0.001/page) | Free | Free |
| Prompts | Same 5 | Same 5 | N/A (ekern output only) |
| Batch mode | Yes (JSONL + resume) | No | No |
| Output | JSON / JSONL | Text | ekern notation |

## References

- LiteLLM: [docs.litellm.ai](https://docs.litellm.ai/)
- Gemini 2.5 Flash: [ai.google.dev](https://ai.google.dev/)
- qwen3-omr sibling: [`../qwen3_vl/`](../qwen3_vl/)
- smt-omr sibling: [`../smt_omr/`](../smt_omr/)
- Sample source: [Library of Congress — Biddle's Piano Waltz (1884)](https://www.loc.gov/item/2023852770/)
