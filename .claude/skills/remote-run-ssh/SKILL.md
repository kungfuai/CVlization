---
name: remote-run-ssh
description: Coordinate remote CVlization workflows on the shared `ssh l1` host—syncing workspaces, bootstrapping environments, running training or evaluation scripts, and collecting logs/artifacts without touching the user’s main checkout.
---

# Remote Run over SSH

Operate CVlization tasks on the remote GPU machine reachable via `ssh l1`. This playbook covers syncing source code to `/tmp`, standing up an isolated Python environment, executing arbitrary scripts (training, evaluation, benchmarks), and retrieving artifacts—all while leaving the user’s long-lived checkout untouched.

## When to Use
- Running heavy trainings or evaluations (e.g., CIFAR10 speed runs, torch pipelines) that need CUDA hardware.
- Measuring regressions (perf, accuracy) on remote GPUs after local code changes.
- Generating reproducible logs/artifacts for discussions or CI baselines.

## Prerequisites
- Local repo state ready to sync (uncommitted changes acceptable).
- SSH config already maps the GPU machine to `l1`.
- Remote host provides CUDA-capable GPU (currently NVIDIA A10) and outbound internet.
- Sufficient remote `/tmp` space (~20 GB) for repo clone, virtualenv, and any runtime caches.

## Quick Reference
1. `rsync` workspace to `/tmp/cvlization_remote` on `l1` (exclude `.git`, `.venv`, heavy caches).
2. Create `/tmp/cvlization_remote/.venv`, upgrade `pip`, install required deps (e.g., torch, tensorflow).
3. Drop helper scripts in `scripts/` (see templates) with `sys.path` bootstrap and any compatibility shims.
4. Execute desired commands with `CUDA_VISIBLE_DEVICES=0` (or other env), redirect stdout to `run_*.log`.
5. Pull logs/artifacts back (or summarize remotely) and update `var/skills/remote-run-ssh/runs/<timestamp>/log.md`.

## Detailed Procedure

### 1. Sync workspace to remote `/tmp`
```bash
rsync -az --delete \
  --exclude='.git' --exclude='var' --exclude='.venv' --exclude='__pycache__' \
  ./ l1:/tmp/cvlization_remote/
```
Keep the canonical remote path `/tmp/cvlization_remote` to ease script reuse. Avoid touching any existing `$HOME/CVlization` checkout owned by the user.

### 2. Bootstrap Python environment
```bash
ssh l1 'python3 -m venv /tmp/cvlization_remote/.venv'
ssh l1 'source /tmp/cvlization_remote/.venv/bin/activate && pip install --upgrade pip'
ssh l1 'source /tmp/cvlization_remote/.venv/bin/activate && \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
```
Install extra dependencies as tasks require (e.g., tensorflow, datasets). When numpy ≥ 2.0 is present, patch legacy aliases (see template).

### 3. Verify GPU availability
```bash
ssh l1 'nvidia-smi'
```
Ensure no conflicting jobs are consuming the GPU. Abide by shared-environment etiquette.

### 4. Materialize runner scripts (example templates)
Create `scripts/run_david_trainer.py` for CIFAR10 training:
```python
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(np, "float"):
    np.float = float  # backwards compat for davidnet backend

import torch
from cvlization.torch.net.image_classification.davidnet.dawn_utils import net, Network
from cvlization.torch.trainer.david_trainer import DavidTrainer


def run(epochs: int = 12, batch_size: int = 512):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network(net()).to(device).half()
    trainer = DavidTrainer(
        model=model,
        epochs=epochs,
        train_batch_size=batch_size,
        use_cached_cifar10=True,
        train_dataset=None,
        val_dataset=None,
    )
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"RESULT_DAVID elapsed_seconds={elapsed:.2f} epochs={epochs} batch_size={batch_size}")


if __name__ == "__main__":
    run()
```

Create `scripts/run_hlb.py`:
```python
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvlization.torch.training_pipeline.image_classification import hlb


def run(train_epochs: float = 11.5):
    hlb.hyp["misc"]["train_epochs"] = train_epochs
    start = time.time()
    ema_val_acc = hlb.main()
    elapsed = time.time() - start
    print(f"RESULT_HLB elapsed_seconds={elapsed:.2f} train_epochs={train_epochs} ema_val_acc={ema_val_acc}")


if __name__ == "__main__":
    run()
```
Create additional scripts for other tasks (evaluation, benchmarks, dataset prep) by following the same `sys.path` and logging patterns.

### 5. Execute runs with logging
```bash
ssh l1 'cd /tmp/cvlization_remote && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_david_trainer.py > run_david.log'

ssh l1 'cd /tmp/cvlization_remote && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_hlb.py > run_hlb.log'
```
Tail logs to monitor progress:
```bash
ssh l1 'cd /tmp/cvlization_remote && tail -f run_david.log'
```
Adapt commands to your script names and environment variables (e.g., huggingface caches, wandb keys). Expect datasets to populate under `/tmp/cvlization_remote/data/` unless overridden.

### 6. Capture metrics
Logs should emit per-epoch tables and a final `RESULT_*` summary. Record:
- Wall-clock elapsed seconds / throughput
- Accuracy, loss, or other task-specific metrics
- Notable warnings (e.g., numpy alias, `torch.load` FutureWarning)

### 7. Retrieve artifacts (optional)
```bash
rsync -az l1:/tmp/cvlization_remote/run_david.log ./remote_runs/$(date +%Y%m%dT%H%M%S)_david.log
rsync -az l1:/tmp/cvlization_remote/run_hlb.log ./remote_runs/$(date +%Y%m%dT%H%M%S)_hlb.log
```
Export checkpoints, TensorBoard logs, or evaluation outputs as needed.

### 8. Document the run
Create `var/skills/remote-run-ssh/runs/<timestamp>/log.md` summarizing inputs, commands, log locations, and outcomes. Mirror the format used by other skills (headers, bullet summaries).

### 9. Optional cleanup
- Remove `/tmp/cvlization_remote` when done if disk pressure exists (`rm -rf` as appropriate).
- Clear cached datasets (`rm -rf /tmp/cifar10`, etc.) only if future runs should start fresh.

## Troubleshooting
- **`ModuleNotFoundError: cvlization`** – ensure scripts prepend repo root to `sys.path`.
- **`AttributeError: np.float`** – confirm the numpy shim is present when using legacy numpy-dependent code.
- **CUDA OOM** – reduce batch size, precision, or run one pipeline at a time.
- **No GPU detected** – verify `CUDA_VISIBLE_DEVICES` assignment and `nvidia-smi` availability.
- **Permission errors** – ensure `/tmp/cvlization_remote` and cache dirs are writable.

## Outputs
Populate the run-log directory and note remote artifact paths. If logs remain on the server, include retrieval instructions in the summary so future users can fetch them.
