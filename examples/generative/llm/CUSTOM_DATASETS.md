# Using Custom Datasets with Unsloth

All Unsloth fine-tuning examples in this repository are controlled via `config.yaml` files, making it easy to use your own datasets without modifying Python code.

## Quick Start

1. **Edit `config.yaml`** in your example directory and update the `dataset` section:

```yaml
dataset:
  path: "your-username/your-dataset"  # HuggingFace dataset or local path
  format: "alpaca"  # Choose: alpaca, sharegpt, or custom
  split: "train"
  max_samples: 1000  # Optional: limit dataset size for testing
```

2. **Run training** as usual:
```bash
bash train.sh
```

## Supported Dataset Formats

### 1. Alpaca Format

Expects three columns: `instruction`, `input`, `output`

**Example JSON:**
```json
[
  {
    "instruction": "Translate to French",
    "input": "Hello world",
    "output": "Bonjour le monde"
  },
  {
    "instruction": "Summarize this text",
    "input": "Long article here...",
    "output": "Brief summary here"
  }
]
```

**Config:**
```yaml
dataset:
  path: "./my_alpaca_data.json"
  format: "alpaca"
```

### 2. ShareGPT Format

Expects a column named `conversations` (or `messages` for some models) containing chat message lists.

**Example JSON (conversations):**
```json
[
  {
    "conversations": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language..."}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "Explain quantum computing"},
      {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
    ]
  }
]
```

**Example JSON (messages for GPT-OSS):**
```json
[
  {
    "messages": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI stands for artificial intelligence..."}
    ]
  }
]
```

**Config:**
```yaml
dataset:
  path: "your-username/chat-dataset"
  format: "sharegpt"
```

### 3. Custom Format

Expects a `text` column with pre-formatted training strings (already in the model's chat template format).

**Example JSON:**
```json
[
  {
    "text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI stands for artificial intelligence..."
  },
  {
    "text": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>"
  }
]
```

**Config:**
```yaml
dataset:
  path: "./my_custom_data.json"
  format: "custom"
```

## Using Local Files

Local files are automatically detected by file extension. Supported formats:
- `.json` - JSON array of objects
- `.jsonl` - Newline-delimited JSON (one object per line)
- `.csv` - CSV with header row
- `.parquet` - Apache Parquet format

**Examples:**

```yaml
# Local JSON file
dataset:
  path: "./data/my_dataset.json"
  format: "alpaca"

# Local JSONL file
dataset:
  path: "./conversations.jsonl"
  format: "sharegpt"

# Local CSV file
dataset:
  path: "./training_data.csv"
  format: "custom"

# Local Parquet file
dataset:
  path: "./data.parquet"
  format: "alpaca"
```

## Using HuggingFace Datasets

Simply specify the dataset name from HuggingFace Hub:

```yaml
dataset:
  path: "yahma/alpaca-cleaned"  # HuggingFace dataset
  format: "alpaca"
  split: "train"  # Specify which split to use
```

## Adjusting Training Parameters

All training hyperparameters are controlled in `config.yaml`:

```yaml
training:
  max_steps: 1000  # Or set to -1 to use num_epochs instead
  num_epochs: 3  # Only used if max_steps == -1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 5
  logging_steps: 1
  save_steps: 100
```

## Model-Specific Notes

### Llama 3.2 3B
- Supports: `alpaca`, `sharegpt`, `custom`
- Uses `conversations` column for ShareGPT

### GPT-OSS 20B
- Supports: `sharegpt`, `custom`
- Uses `messages` column for ShareGPT (not `conversations`)
- Requires smaller batch sizes due to model size

### Qwen 2.5 7B
- Supports: `alpaca`, `sharegpt`, `custom`
- Uses `conversations` column for ShareGPT
- Supports up to 40k context length

## Testing Your Dataset

Each example includes a `test.sh` smoke test. To test with your custom data:

1. Create a small test file (e.g., 100 samples)
2. Update `config.yaml` to point to your test file
3. Run the smoke test:
```bash
bash test.sh
```

This will train for 2 steps to verify your data loads correctly.

## Troubleshooting

**Error: "Custom format requires a 'text' column"**
- Make sure your dataset has a column named `text` when using `format: "custom"`

**Error: "Unsupported file format"**
- Only `.json`, `.jsonl`, `.csv`, `.parquet` are supported for local files

**Error: "KeyError: 'instruction'"**
- When using `format: "alpaca"`, ensure your dataset has `instruction`, `input`, `output` columns

**Error: "KeyError: 'conversations'" or "KeyError: 'messages'"**
- Check which column name your model expects (see Model-Specific Notes above)

## Example Workflow

1. **Prepare your data** in one of the supported formats
2. **Test locally** with a small subset:
```yaml
dataset:
  path: "./test_data.json"
  format: "alpaca"
  max_samples: 100
```
3. **Run smoke test**: `bash test.sh`
4. **Scale up** by removing `max_samples` and increasing `max_steps`
5. **Monitor training** by watching loss decrease in the logs
