# SAM LoRA Fine-tuning

Fine-tune [SAM ViT-B](https://github.com/facebookresearch/segment-anything) with LoRA adapters on COCO-format or image+mask segmentation datasets. Uses DiceCELoss from MONAI for stable, NaN-free training.

## Requirements

- Docker with NVIDIA GPU support
- ~10 GB VRAM (single GPU)

## Quick start

```bash
# Build the Docker image
bash build.sh

# Train on the built-in ring dataset (downloads automatically)
bash train.sh --epochs 10

# Train on a HuggingFace dataset
bash train.sh --hf-dataset keremberke/pcb-defect-segmentation --epochs 5

# Train on a local dataset
bash train.sh --dataset-dir /path/to/coco/dataset --epochs 20
```

Or via `cvl`:

```bash
cvl run sam_lora_finetuning build
cvl run sam_lora_finetuning train -- --epochs 10
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-dataset` | | HuggingFace dataset to download |
| `--dataset-dir` | | Path to local COCO or image+mask dataset |
| `--output-dir` | `outputs` | Directory for checkpoints and logs |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `1` | Batch size |
| `--rank` | `512` | LoRA rank |
| `--lr` | `1e-4` | Learning rate |
| `--checkpoint` | | Path to SAM ViT-B checkpoint (auto-downloads if omitted) |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb-project` | `sam-lora-finetuning` | W&B project name |
| `--wandb-run-name` | | W&B run name |

## Dataset formats

Two formats are auto-detected:

**COCO JSON** -- expected layout:
```
dataset/
  train/
    _annotations.coco.json
    images/
  valid/          # optional
    _annotations.coco.json
    images/
```

**Image + Mask** -- expected layout:
```
dataset/
  train/
    images/
    masks/        # matched by filename stem
  test/           # optional, used for validation
    images/
    masks/
```

## Project structure

```
src/
  config/base_config.py          # Pydantic BaseSettings with CLI parsing
  training/training_config.py    # All training hyperparameters
  training/trainer.py            # Train/validate/visualize logic
  training/training_session.py   # Orchestrator (entry point)
  data/dataset.py                # Dataset classes and download helpers
  data/dataset_builder.py        # Builds dataloaders from config
  models/lora.py                 # LoRA adapter injection
  models/sam_forward.py          # Gradient-enabled SAM forward pass
```

## Outputs

- `outputs/lora_rank<R>.safetensors` -- final LoRA weights
- `outputs/lora_rank<R>_best.safetensors` -- best validation loss checkpoint (when validation data is available)

## Export

```bash
cvl export sam_lora_finetuning -o ./my-project
cd my-project
bash bin/build.sh
bash bin/train.sh --epochs 5
```

Generates a standalone project with Docker Compose scaffolding.
