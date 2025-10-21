# Modded-NanoGPT - Docker Setup for CVlization

This folder contains a self-contained, Dockerized version of the modded-nanogpt repository, configured to work with CVlization infrastructure.

## Quick Start

### 1. Build the Docker Image
```bash
./build.sh
```

This builds a Docker image based on `cvlization-torch-gpu` with all required dependencies including:
- PyTorch 2.6.0 nightly (for FlexAttention support)
- numpy, tqdm, huggingface-hub
- tiktoken, einops, wandb

### 2. Train the Model

Basic training (using all available GPUs):
```bash
./train.sh
```

Custom training options:
```bash
# Use 4 GPUs instead of 8
./train.sh --num-gpus 4

# Enable Weights & Biases logging
./train.sh --wandb

# Specify data directory
./train.sh --data-dir /path/to/data

# Memory optimization for A10 GPU (23GB):
./train.sh --train-seq-len 16384 --val-seq-len 32768 --num-gpus 1 --max-batch-span-multiplier 4 --train-loss-every 10

# Higher memory usage (adjust based on your GPU):
./train.sh --train-seq-len 32768 --val-seq-len 65536 --num-gpus 1
```

The training script will:
- Automatically download the FineWeb dataset (1B tokens) if not present
- Create directories for checkpoints and logs
- Run training using PyTorch distributed (torchrun)
- Save checkpoints to `./checkpoints/`
- Save logs to `./logs/`

### 3. Generate Text

After training, use the generation script:
```bash
# Use default checkpoint
./generate.sh

# Specify checkpoint and prompt
./generate.sh --checkpoint checkpoints/model_100000.pt --prompt "The future of AI is"

# Adjust generation parameters
./generate.sh --max-tokens 512 --temperature 0.9 --top-k 40
```

## Directory Structure
```
modded-nanogpt/
├── Dockerfile          # Docker configuration
├── build.sh           # Build Docker image
├── train.sh           # Training wrapper script
├── generate.sh        # Generation script
├── data/              # Training data (auto-downloaded)
├── checkpoints/       # Model checkpoints
├── logs/              # Training logs
├── train_gpt.py       # Main training code (3-minute speedrun)
├── train_gpt_medium.py # Medium model variant
└── records/           # Historical speedrun records
```

## Key Features

- **Optimized for Speed**: Achieves GPT-2 quality (3.28 val loss) in just 3 minutes on 8xH100
- **Docker Integration**: Fully containerized with cvlization-torch-gpu base image
- **Flexible GPU Support**: Configurable from 1 to 8 GPUs
- **Automatic Data Download**: Downloads FineWeb dataset on first run
- **Modern Architecture**: Includes RoPE, QK-Norm, ReLU², and FlexAttention
- **Muon Optimizer**: Custom optimizer for faster convergence

## Environment Variables

The Docker container uses these environment variables:
- `CUDA_VISIBLE_DEVICES`: Controls which GPUs to use
- `NCCL_P2P_DISABLE=1`: Disables P2P for better compatibility
- `PYTHONUNBUFFERED=1`: Ensures real-time output
- `WANDB_MODE`: Controls Weights & Biases logging (disabled by default)

## Troubleshooting

1. **CUDA/NCCL Version Issues**: The Docker image handles this automatically
2. **Out of Memory**: Reduce train-seq-len (which controls both sequence length and effective batch size)
3. **Slow First Run**: torch.compile adds ~5 minutes on first execution
4. **Data Download**: First run downloads 1GB of training data

## Advanced Usage

### Running Inside Container
```bash
docker run -it --rm --gpus all -v $(pwd):/workspace modded-nanogpt bash
# Then inside container:
python data/cached_fineweb10B.py 10  # Download more data
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Custom Training Script
Modify `train_gpt.py` parameters directly for advanced configurations.

## Original Repository

This is based on: https://github.com/KellerJordan/modded-nanogpt

The original repo contains the full NanoGPT speedrun history and contributions from many researchers.
