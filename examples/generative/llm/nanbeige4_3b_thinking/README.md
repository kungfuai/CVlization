# Nanbeige4-3B-Thinking Inference (Dockerized)

Lightweight inference example for the Nanbeige4-3B-Thinking-2510 reasoning model. Supports chat and tool-call prompts via the model's built-in chat template and special tokens (`<think>`, `<tool_call>`, `<|im_end|>`).

## Files
- `predict.py` — CLI for chat/tool inference with truncation guard and CVL dual-mode paths.
- `Dockerfile`, `requirements.txt`, `build.sh` — Build a CUDA runtime image (PyTorch 2.5.1, Transformers 4.46).
- `predict.sh`, `test.sh` — Run inference in Docker with HF cache mount; smoke prompt example.
- `example.yaml` — CVL metadata.
- Smoke test uses `--max-new-tokens 512` to ensure the model finishes its think block and returns a final answer.
- Predict preset verified via `cvl run ... predict -- --prompt "Tell me a fun fact about prime numbers under 20." --max-new-tokens 128 --temperature 0.0` (A10 GPU).

## Requirements
- GPU with ≥12 GB VRAM recommended (bf16 on CUDA compute ≥8.0; falls back to fp16/fp32).
- Hugging Face access to `Nanbeige/Nanbeige4-3B-Thinking-2510`; safetensors pulled at first run.
- Chat EOS token is `166101` (`<|im_end|>`); pad defaults to EOS if unset.

## Quickstart
```bash
cd examples/generative/llm/nanbeige4_3b_thinking
./build.sh

# Simple chat
./predict.sh --prompt "Which number is bigger, 9.11 or 9.8?" --max-new-tokens 128
```

To avoid unnecessary sampling noise or to force a full answer (model often writes a <think> block first):
```bash
./predict.sh --prompt "Summarize the Nanbeige4-3B-Thinking model." \
  --temperature 0.0 --max-new-tokens 256
```

## Tool Call Example
Create `tools.json`:
```json
[
  {
    "type": "function",
    "function": {
      "name": "SearchWeather",
      "description": "Find current weather in a Chinese city.",
      "parameters": {
        "type": "object",
        "properties": { "location": { "type": "string" } },
        "required": ["location"]
      }
    }
  }
]
```
Then run:
```bash
./predict.sh \
  --prompt "Help me check the weather in Beijing now." \
  --tools-json tools.json \
  --max-new-tokens 128
```

## Notes
- Context window is 65k in config, but truncate as needed with `--truncate-input-tokens` to control KV cache growth.
- The model tends to emit a `<think>` block; `predict.py` strips it and prints the final answer. Use `--max-new-tokens 512` if you need the model to fully close the think block and give a final reply (smoke test uses 512).
- `trust_remote_code` is optional (default off); the model uses standard Llama architecture and tokenizer.
- Outputs save to `outputs/` by default; use `--output-file` to override (e.g., `--output-file smoke_test.txt` writes to `outputs/smoke_test.txt`).
