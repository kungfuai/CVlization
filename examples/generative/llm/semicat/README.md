# Semicat: Categorical Flow Maps

Train a categorical flow model for discrete text generation.

Semicat learns continuous flow maps over discrete token spaces, enabling non-autoregressive text generation. The model interpolates between a Gaussian prior and one-hot encoded text, trained with a vector field matching (VFM) loss and optional self-distillation (SD).

## Quick start

```bash
cvl run semicat build
cvl run semicat test       # smoke test (~15s)
cvl run semicat train      # full training on Text8
```

## Datasets

Use `--dataset` / `-d` to choose a dataset:

```bash
# Text8 (default) — char-level, 27 vocab, 100MB
cvl run semicat train

# Custom text file — char-level, 27 vocab (a-z + space)
cvl run semicat train -- -d /path/to/my_corpus.txt

# LM1B — BERT tokenizer, 30K vocab, 128 seq len
cvl run semicat train -- -d lm1b

# OpenWebText — GPT2 tokenizer, 50K vocab, 1024 seq len
cvl run semicat train -- -d openwebtext
```

For custom text files, the text is lowercased and normalized to 27 characters (a-z + space). The file is split 90/10 for train/val.

## Options

```
--dataset, -d     Dataset: text8, lm1b, openwebtext, or path to .txt file
--tokenizer       Tokenizer for HF datasets: gpt2 (default) or bert
--seq-len         Sequence length (default: 256 char, 128 lm1b, 1024 owt)
--batch-size      Batch size
--steps           Training steps
--compile         Enable torch.compile
```

Extra arguments are passed as Hydra overrides:

```bash
# Adjust learning rate
cvl run semicat train -- model.optimizer.lr=5e-5

# Use larger model
cvl run semicat train -- experiment=text8_dit_lg

# More frequent validation (generates text samples + metrics)
cvl run semicat train -- trainer.val_check_interval=1000

# Enable NLL computation during validation (slow but informative)
cvl run semicat train -- model.calc_nll=true
```

## WandB logging

During validation, the model generates text samples using different numbers of ODE sampling steps (1, 2, 4, 8, 16). More steps = better quality. These appear in WandB as:

- `samples/001` through `samples/016` — generated text tables (number = sampling steps)
- `val/entropy_N_steps` — token entropy of generated samples
- `train/loss`, `val/loss` — training and validation loss curves

Samples are logged at each validation checkpoint, so you can track text quality improving over training.

```bash
# Set your API key (or export in shell)
export WANDB_API_KEY=your_key

# Train with wandb logging
cvl run semicat train -- logger=wandb

# Combine with other options
cvl run semicat train -- -d lm1b logger=wandb
```

## Architecture

- **DUO**: Diffusion Transformer with RoPE, adaptive LayerNorm, and Flash Attention
- **28.6M params** (text8_dit_sm), configurable depth/width
- **bf16 mixed precision** training

## Resources

| Dataset | VRAM | Steps | Time estimate |
|---|---|---|---|
| Text8 (sm) | ~8 GB | 100K | Hours (1 GPU) |
| Text8 (lg) | ~16 GB | 500K | Days (1 GPU) |
| LM1B (bs=64) | ~12 GB | 120K | ~4 hours (1 GPU, ~8 steps/s) |
| OpenWebText | ~24+ GB | 250K | Days (multi-GPU) |

## References

- Paper: https://arxiv.org/abs/2602.12233
- Code: https://github.com/olsdavis/semicat
- Dataset: http://mattmahoney.net/dc/textdata.html
