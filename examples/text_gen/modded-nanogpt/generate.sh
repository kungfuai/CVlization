#!/bin/bash

# Generation/Inference script for modded-nanogpt

# Default values
CHECKPOINT=${CHECKPOINT:-"checkpoints/latest.pt"}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_K=${TOP_K:-50}
PROMPT=${PROMPT:-"Once upon a time"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./generate.sh [--checkpoint PATH] [--max-tokens N] [--temperature T] [--top-k K] [--prompt TEXT]"
      exit 1
      ;;
  esac
done

echo "Running generation with:"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Max tokens: $MAX_TOKENS"
echo "  - Temperature: $TEMPERATURE"
echo "  - Top-k: $TOP_K"
echo "  - Prompt: '$PROMPT'"

# Create a simple generation script if it doesn't exist
if [ ! -f "generate.py" ]; then
  cat > generate.py << 'EOF'
import torch
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_gpt import GPT, GPTConfig

def generate_text(checkpoint_path, prompt, max_tokens=256, temperature=0.8, top_k=50):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model config from checkpoint
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)

    # Initialize model
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Fix state dict keys if needed
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()

    # Simple tokenization (for demonstration)
    # In production, use proper tokenizer
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device='cuda')[None, ...]

    # Generate
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits
            logits = model(x)[0]
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            x = torch.cat((x, idx_next), dim=1)

    # Decode
    tokens = x[0].tolist()
    text = enc.decode(tokens)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="Once upon a time")
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    args = parser.parse_args()

    result = generate_text(
        args.checkpoint,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k
    )
    print(result)
EOF
fi

# Run the Docker container for generation
docker run -it --rm \
  --gpus all \
  -v "$(pwd)/checkpoints:/workspace/checkpoints" \
  -v "$(pwd)/generate.py:/workspace/generate.py" \
  modded-nanogpt \
  python generate.py \
    --checkpoint "$CHECKPOINT" \
    --prompt "$PROMPT" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K"