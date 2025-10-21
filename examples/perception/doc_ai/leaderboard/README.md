# Doc AI Leaderboard

Simple benchmark tool to compare different document AI inference recipes on the same test images.

## Directory Structure

```
leaderboard/
├── config.json              # Test images and model configurations
├── run_benchmark.sh         # Main benchmark runner
├── compare_outputs.py       # Compare outputs side-by-side
├── test_data/              # Test images (add your own)
└── results/                # Benchmark results (timestamped)
    └── YYYYMMDD_HHMMSS/
        ├── benchmark.csv   # Timing and status results
        ├── run_info.txt    # Run metadata
        ├── moondream2/     # Model-specific outputs
        ├── granite-docling/
        └── ...
```

## Quick Start

### 1. Add Test Images

```bash
mkdir -p test_data
cp /path/to/your/test/images/*.png test_data/
```

### 2. Configure Models

Edit `config.json` to add test images and configure which models to run:

```json
{
  "test_images": [
    "test_data/invoice.png",
    "test_data/receipt.jpg"
  ],
  "models": {
    "moondream2": {
      "script": "../moondream2/predict.sh",
      "args": "--task ocr"
    },
    "granite-docling": {
      "script": "../granite-docling/predict.sh",
      "args": "--format markdown"
    }
  }
}
```

### 3. Run Benchmark

```bash
# Run all models from config
./run_benchmark.sh

# Run specific models only
./run_benchmark.sh moondream2 granite-docling

# Run single model
./run_benchmark.sh moondream2
```

### 4. View Results

```bash
# List all runs
ls -lt results/

# View benchmark CSV
cat results/20250116_123456/benchmark.csv

# View specific model output
cat results/20250116_123456/moondream2/invoice_output.txt

# Compare all outputs side-by-side
python compare_outputs.py results/20250116_123456/

# Export comparison as markdown
python compare_outputs.py results/20250116_123456/ --format markdown > comparison.md
```

## Output Format

### benchmark.csv

```csv
model,image,time_seconds,status
moondream2,invoice.png,2.34,success
granite-docling,invoice.png,1.89,success
moondream2,receipt.jpg,2.12,success
granite-docling,receipt.jpg,1.95,success
```

### Directory Layout

```
results/20250116_143022/
├── benchmark.csv
├── run_info.txt
├── moondream2/
│   ├── invoice_output.txt
│   ├── invoice_log.txt
│   ├── receipt_output.txt
│   └── receipt_log.txt
└── granite-docling/
    ├── invoice_output.txt
    ├── invoice_log.txt
    ├── receipt_output.txt
    └── receipt_log.txt
```

## Adding New Models

1. Add model entry to `config.json`:

```json
{
  "models": {
    "your-model": {
      "script": "../your-model/predict.sh",
      "args": "--your-args"
    }
  }
}
```

2. Ensure your model's `predict.sh` accepts:
   - First argument: input image path
   - `--output` flag: output file path
   - Any additional args specified in config

Example predict.sh signature:
```bash
./predict.sh <input_image> --output <output_file> [additional_args]
```

## Tips

- Keep test images small (< 5MB) for faster iteration
- Use descriptive image names (e.g., `invoice_handwritten.png`)
- Save important benchmark results with descriptive names:
  ```bash
  mv results/20250116_143022 results/baseline_v1
  ```
- Track results in git (optional):
  ```bash
  git add results/baseline_v1/benchmark.csv
  git commit -m "Baseline benchmark results"
  ```

## Analyzing Results

### Find fastest model per image

```bash
# Using awk
awk -F',' 'NR>1 {print $2,$1,$3}' results/latest/benchmark.csv | sort -k1,1 -k3,3n | awk '{if($1!=p){print; p=$1}}'
```

### Calculate average inference time per model

```bash
# Using awk
awk -F',' 'NR>1 && $4=="success" {sum[$1]+=$3; count[$1]++} END {for(m in sum) printf "%s: %.2fs\n", m, sum[m]/count[m]}' results/latest/benchmark.csv
```

### Compare output lengths

```bash
for dir in results/latest/*/; do
    model=$(basename "$dir")
    echo "$model:"
    for f in "$dir"/*_output.txt; do
        wc -c "$f" | awk '{print "  " $2 ": " $1 " chars"}'
    done
done
```
