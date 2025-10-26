## Quickstart

```
bash examples/generative/llm/mistral7b/build.sh
# quick smoke test running a single optimisation step
bash examples/generative/llm/mistral7b/train.sh --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --no_load_in_4bit --max_steps 1 --max_train_samples 8 --max_eval_samples 4 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --eval_steps 1 \
    --save_steps 1 --logging_steps 1 --do_eval False

# original full run (requires HF access to mistralai/Mistral-7B-v0.1)
bash examples/generative/llm/mistral7b/train.sh
```

All training options are now configurable through command-line flags; run
`python train.py --help` for the full list (model ID, dataset limits, LoRA
hyperparameters, evaluation cadence, etc.).

The original settings take roughly 2 hours on an AWS g5 instance to finish
1000 steps when using the full 7B base model.

If the GEM/VIGGO dataset is unavailable (e.g. no Hugging Face token), the
script automatically falls back to the tiny sample JSONs bundled in
`data/`. You can also point to your own data by supplying
`--train_file/--eval_file` with any JSON/CSV/Parquet source.

You need to log in to huggingface and request access to the pretrained model, to avoid the following error:

```
OSError: You are trying to access a gated repo.
```

## Reference

This is adapted from https://brev.dev/blog/fine-tuning-mistral
