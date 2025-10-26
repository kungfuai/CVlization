## Quickstart

```bash
bash examples/generative/llm/mixtral8x7b/build.sh

# quick smoke test (4-bit, short completion)
bash examples/generative/llm/mixtral8x7b/predict.sh \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --no_load_in_4bit \
  --prompt "Summarise the product launch of Mixtral." \
  --max_new_tokens 64 --temperature 0.7 --top_p 0.9 \
  --output_file outputs/summary.txt

# switch to full precision (useful for small GPUs or alternative models)
bash examples/generative/llm/mixtral8x7b/predict.sh \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --no_load_in_4bit --prompt "Write a haiku about GPUs." --max_new_tokens 32

# run the original Mixtral 8x7B MoE (requires HF token and ~20GB VRAM)
bash examples/generative/llm/mixtral8x7b/predict.sh \
  --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --load_in_4bit --prompt "Draft a positive status update about our release." \
  --max_new_tokens 128 --temperature 0.7
```

`predict.py` exposes the usual generation knobs (temperature, top-*p/k*,
repetition penalty, output paths, etc.). Run `python predict.py --help` for the
full list of options.

You need to log in to Hugging Face and request access to the pretrained model to avoid the following error:

```
OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
```

The CLI no longer hard-codes offloading requirements. By default we run in
full precision (ideal for small open models such as TinyLlama). Add
`--load_in_4bit` to activate 4-bit quantization when `bitsandbytes` is
available.
