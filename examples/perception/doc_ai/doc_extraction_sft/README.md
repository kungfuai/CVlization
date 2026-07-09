# Document Extraction SFT

Fine-tune an instruction LLM for post-OCR document extraction into structured outputs.

This example is for text-in/text-out DocAI post-processing:

```text
PDF/images -> OCR text -> schema-aware prompt -> LLM extraction model -> structured output
```

It is intentionally domain-neutral. The same trainer can be used for forms, invoices, correspondence, applications, contracts, or other document packages as long as each training row provides chat messages and a structured target. JSON is the common target format, but the SFT path can train on other serialized targets such as XML, YAML, Markdown tables, or line-based key-value text.

## Dataset Contract

Training data is loaded with `datasets.load_dataset`. The default configs expect a local JSONL file:

```yaml
dataset:
  path: "json"
  data_files: "$DOC_EXTRACTION_SFT_TRAIN_JSONL"
  split: "train"
```

You can either set that environment variable, put a literal path in a config file, or pass the dataset at runtime with `--data-files`.

Each row should contain:

- `messages`: chat messages that include the instruction, OCR text, and schema context.
- `target_json` or `target_text`: the assistant response to train on.
- `target_source`: label for filtering rows, usually `ground_truth`; optional distilled rows can use a project-defined value such as `teacher`.
- `document_id`: stable grouping key used to keep related rows out of both train and eval.

Optional teacher outputs, quality warnings, and task-specific metadata are preserved for project-owned filtering and evaluation scripts.

Set `dataset.target_column` when the target is not in `target_json`:

```yaml
dataset:
  target_column: "target_text"
```

If ground-truth field objects contain metadata like `value`, `comparator`, `normalizer`, `page`, or `path`, set:

```yaml
ground_truth_target_format: "value_only"
transform_ground_truth_schema: true
```

That trains on the extracted values rather than evaluator metadata. If your target is already compact, set `transform_ground_truth_schema: false`.

## Quick Start

```bash
cvl run doc_extraction_sft build
cvl run doc_extraction_sft test
```

`./test.sh` uses the committed fixture at `fixtures/smoke.jsonl`, runs two training steps, and defaults to one GPU with `CVL_GPUS=0`.

To train on your own dataset:

```bash
CVL_IMAGE=doc_extraction_sft CVL_GPUS=0 \
  cvl run doc_extraction_sft train -- \
    --config config.yaml \
    --data-files /path/to/train.jsonl
```

Validate a local dataset before training:

```bash
cvl run --no-docker doc_extraction_sft validate \
  -- \
  --config config.yaml \
  --data-files /path/to/train.jsonl
```

Useful environment variables:

```bash
export HF_TOKEN=...
export CVL_HF_CACHE=/path/to/huggingface-cache
export CVL_IMAGE=doc_extraction_sft
export CVL_GPUS=0,1
```

If `CVL_GPUS` is unset, `train.sh` exposes all GPUs to Docker.

## Runtime Profiles

This example has several Dockerfiles because some model families need different
dependency stacks. Use `--runtime-profile` instead of remembering image names:

| Profile | Image | Use case |
|---|---|---|
| `default` | `doc_extraction_sft` | Baseline Transformers/TRL stack |
| `modern` | `doc_extraction_sft_modern` | Newer Transformers stack with FlashAttention |
| `qwen35` | `doc_extraction_sft_qwen35` | Qwen3.5 Transformers experiments |
| `unsloth-qwen35` | `doc_extraction_sft_unsloth_qwen35` | Qwen3.5 Unsloth experiments |
| `unsloth-latest` | `doc_extraction_sft_unsloth_latest` | General Unsloth path for Llama, Qwen, and Ministral-like models |
| `nemotron` | `doc_extraction_sft_unsloth_nemotron` | Nemotron/Mamba-specific Unsloth stack |
| `gemma4` | `doc_extraction_sft_gemma4` | Gemma 4 Transformers experiments |
| `unsloth-gemma4` | `doc_extraction_sft_unsloth_gemma4` | Gemma 4 Unsloth experiments |

Build a profile:

```bash
cvl run doc_extraction_sft build -- --runtime-profile nemotron
```

Run train/eval with the same profile:

```bash
cvl run doc_extraction_sft train-unsloth -- \
  --runtime-profile nemotron \
  --config config_unsloth_nemotron3_nano_4b_multitask_8k_smoke.yaml \
  --data-files /path/to/train.jsonl
```

The profile flag selects the Docker image before the trainer starts and is not
passed through to `train_unsloth.py`. The older `build-*` presets still work as
aliases for compatibility.

## Model Configs

The default config uses Qwen2.5-7B with LoRA/QLoRA, assistant-only loss masking, and a 32k context window:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_length: 32768
  max_target_length: 8192
```

Additional configs cover Qwen3, Qwen3.5, Llama 3.1, Phi-4, Gemma, and GPT-OSS experiments. Most are meant to be run against a JSONL dataset passed through `--data-files` or configured as `dataset.data_files`.

For the newer Transformers stack:

```bash
cvl run doc_extraction_sft build -- --runtime-profile modern
CVL_GPUS=0 cvl run doc_extraction_sft train -- \
  --runtime-profile modern \
  --config config_llama31_8b_32k_local_smoke.yaml \
  --data-files /path/to/train.jsonl
```

For Qwen3.5:

```bash
cvl run doc_extraction_sft build -- --runtime-profile qwen35
CVL_GPUS=0 cvl run doc_extraction_sft train -- \
  --runtime-profile qwen35 \
  --config config_qwen35_9b_16k_local_smoke.yaml \
  --data-files /path/to/train.jsonl
```

For Unsloth:

```bash
cvl run doc_extraction_sft build -- --runtime-profile unsloth-latest
CVL_GPUS=0 cvl run doc_extraction_sft train-unsloth -- \
  --runtime-profile unsloth-latest \
  --config config_unsloth_llama31_8b_multitask_32k_smoke.yaml \
  --data-files /path/to/train.jsonl
```

## Evaluation

`eval_checkpoint.py` is the preferred entrypoint. It can evaluate a base/full
checkpoint by model id or local path, or evaluate a LoRA adapter on top of the
model in the config. Primary metrics exclude checkbox fields by default because
OCR text often does not reliably encode checkbox state. Use `--exclude-path-regex`
for domain-specific grids or fields that should not count as text extraction. A secondary
`*.all-fields.metrics.json` file is also written unless disabled.

Evaluate a base checkpoint:

```bash
CVL_GPUS=0 cvl run doc_extraction_sft eval -- \
  --runtime-profile modern \
  --config config_qwen3_8b.yaml \
  --data-files /path/to/heldout.jsonl \
  --checkpoint Qwen/Qwen3-8B \
  --name qwen3_8b_base \
  --output-dir outputs/eval/qwen3_8b_base \
  --max-samples 32
```

Evaluate a LoRA adapter:

```bash
CVL_GPUS=0 cvl run doc_extraction_sft eval -- \
  --runtime-profile modern \
  --config config_qwen3_8b.yaml \
  --data-files /path/to/heldout.jsonl \
  --adapter outputs/qwen3-8b-document-extraction-sft/final_model \
  --name qwen3_8b_sft \
  --output-dir outputs/eval/qwen3_8b_sft \
  --max-samples 32
```

Evaluate an Unsloth LoRA adapter:

```bash
CVL_GPUS=0 cvl run doc_extraction_sft eval-unsloth -- \
  --runtime-profile unsloth-latest \
  --config config_unsloth_ministral3_3b_multitask_32k_300.yaml \
  --data-files /path/to/heldout.jsonl \
  --adapter outputs/unsloth-ministral3-3b-document-extraction-sft-multitask-32k-300/checkpoint-100 \
  --name ministral3_3b_sft_step100 \
  --output-dir outputs/eval/ministral3_3b_sft_step100 \
  --max-samples 32
```

When running eval in Docker, pass adapter paths as paths visible inside the
example container. For checkpoints under this example, that usually means
`outputs/.../checkpoint-N` rather than a host path like
`examples/perception/doc_ai/doc_extraction_sft/outputs/...`.

Score an existing predictions JSONL without generating again:

```bash
CVL_NO_DOCKER=1 ./eval.sh \
  --predictions outputs/eval/qwen3_8b_predictions.jsonl \
  --name qwen3_8b_predictions \
  --output-dir outputs/eval/qwen3_8b_predictions
```

To score all fields, including checkboxes:

```bash
python3 evaluate_predictions.py \
  outputs/eval/qwen3_8b_predictions.jsonl \
  --output outputs/eval/qwen3_8b_all_fields.metrics.json \
  --repair-json-string-newlines \
  --normalize-values \
  --include-checkbox-fields
```

## Notes

This example uses `AutoModelForCausalLM` plus tokenizer chat templates. It does not use image prompting; OCR text and schema instructions are already in the prompt.
