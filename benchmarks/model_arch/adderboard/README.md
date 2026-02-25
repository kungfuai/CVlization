# AdderBoard Benchmark (Model Architecture)

Minimal benchmark for evaluating autoregressive transformer submissions on 10-digit integer addition.

This benchmark is adapted from the public AdderBoard challenge and packaged for CVlization benchmarking workflows.

## Task

Build a small autoregressive transformer that computes:

- Input: two integers `a`, `b` in `[0, 9_999_999_999]`
- Output: integer sum `a + b`

Qualification target:

- Accuracy `>= 99%` on:
  - 10 fixed edge cases
  - 10,000 random pairs (`seed=2025`)

## Submission Interface

Each submission must be a Python file exposing:

1. `build_model() -> (model, metadata)`
2. `add(model, a: int, b: int) -> int`

`metadata` should contain:

- `name`
- `author`
- `params` (unique parameter count)
- `architecture`
- `tricks` (list of strings)

Use [submission_template.py](/home/zsi/projects/worktrees/CVlization/model-arch/benchmarks/model_arch/adderboard/submission_template.py) as the starting point.

## Verify One Submission

```bash
cd benchmarks/model_arch/adderboard
python verify.py /path/to/your_submission.py
```

Optional flags:

```bash
python verify.py your_submission.py --num-tests 10000 --seed 2025
```

## Compare Multiple Submissions

```bash
cd benchmarks/model_arch/adderboard
python run_benchmark.py submissions/*.py
```

This creates timestamped results under `results/<timestamp>/`:

- `scores.csv`
- `leaderboard.md`
- `summary.json`

And updates:

- `results/latest` (symlink)

## Build and Run

```bash
cd benchmarks/model_arch/adderboard
bash build.sh
bash predict.sh submissions/reference_python_add.py
```

Docker:

```bash
cd benchmarks/model_arch/adderboard
docker build -t cvl-adderboard-benchmark .
docker run --rm cvl-adderboard-benchmark python verify.py submissions/reference_python_add.py
```

## Notes

- This harness validates task correctness and reports metrics.
- It does not attempt to automatically prove transformer-rule compliance. Rule checks remain part of human review.
