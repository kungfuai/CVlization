#!/usr/bin/env python3
"""
Thin wrapper to launch PipelineRL GRPO training for QED-Nano.

Delegates to `python -m pipelinerl.launch` with the rl_cvl.yaml config and
applies environment-variable-driven overrides for GPU allocation, model,
and grader API endpoint.

PipelineRL orchestrates all components internally on a single node:
  - Actor LLM servers  (vLLM, ACTOR_GPUS GPUs)
  - Rollout actor workers
  - Preprocessor
  - Trainer            (DeepSpeed ZeRO-3, TRAINER_GPUS GPUs)
  - Proof grader       (external OpenAI-compatible API)

Requirements:
  - 4+ NVIDIA GPUs (2 actor + 2 trainer minimum)
  - OPENAI_API_KEY  (or OPENAI_BASE_URL pointing to a local vLLM server)

Env vars (all optional, have defaults):
  MODEL_ID          Base model to train  (default: Qwen/Qwen3-4B-Instruct-2507)
  MODEL_REVISION    HF revision          (default: main)
  ACTOR_GPUS        GPUs for vLLM actor  (default: 2)
  TRAINER_GPUS      GPUs for DeepSpeed   (default: 2)
  SEQ_LENGTH        Max sequence length  (default: 8192)
  GRADER_MODEL      Grader model name    (default: gpt-4o-mini)
  OPENAI_API_KEY    API key for grader
  OPENAI_BASE_URL   Grader server URL    (default: https://api.openai.com/v1)
  WANDB_API_KEY     Enable W&B logging
  OUTPUT_DIR        Checkpoint dir       (default: outputs/rl_<timestamp>)
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PIPELINERL_ROOT = Path(os.environ.get("PIPELINERL_ROOT", "/opt/qed-nano/training"))
CONF_DIR = PIPELINERL_ROOT / "conf"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch PipelineRL GRPO training for QED-Nano.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Minimal run (4 GPUs, OpenAI grader)
  OPENAI_API_KEY=sk-... python train_rl.py

  # 8 GPUs, local vLLM grader
  ACTOR_GPUS=4 TRAINER_GPUS=4 \\
  GRADER_MODEL=Qwen/Qwen3-4B-Instruct-2507 \\
  OPENAI_BASE_URL=http://localhost:8000/v1 \\
  OPENAI_API_KEY=token \\
  python train_rl.py

  # Start from a previous SFT checkpoint
  MODEL_ID=outputs/sft OPENAI_API_KEY=sk-... python train_rl.py
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipelinerl.launch command without executing it.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides passed directly to pipelinerl.launch.",
    )
    args = parser.parse_args()

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    model_rev = os.environ.get("MODEL_REVISION", "main")
    actor_gpus = int(os.environ.get("ACTOR_GPUS", "2"))
    trainer_gpus = int(os.environ.get("TRAINER_GPUS", "2"))
    seq_length = int(os.environ.get("SEQ_LENGTH", "8192"))
    grader_model = os.environ.get("GRADER_MODEL", "gpt-4o-mini")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.environ.get("OUTPUT_DIR", f"outputs/rl_{timestamp}")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY is not set.\n"
            "The proof grader connects to an OpenAI-compatible API.\n"
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL for a local server).",
            file=sys.stderr,
        )

    hydra_overrides = [
        f"model_path={model_id}",
        f"model_revision={model_rev}",
        f"output_dir={output_dir}",
        f"world.actor_fraction={actor_gpus}",
        f"world.finetune_fraction={trainer_gpus}",
        f"finetune.seq_length={seq_length}",
        f"llm.parameters.max_tokens={seq_length}",
        f"test_llm.parameters.max_tokens={seq_length}",
        f"vllm_config.vllm_kwargs.max-model-len={seq_length}",
        f"llm_grader.name={grader_model}",
        *args.overrides,
    ]

    cmd = [
        sys.executable,
        "-m",
        "pipelinerl.launch",
        f"--config-path={CONF_DIR}",
        "--config-name=rl_cvl",
        *hydra_overrides,
    ]

    if args.dry_run:
        print("Command:")
        print(" \\\n  ".join(cmd))
        return 0

    print(f"Model      : {model_id} @ {model_rev}")
    print(f"Output     : {output_dir}")
    print(f"GPU split  : {actor_gpus} actor + {trainer_gpus} trainer = {actor_gpus + trainer_gpus} total")
    print(f"Seq length : {seq_length}")
    print(f"Grader     : {grader_model}")
    print(f"Grader URL : {os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print()

    env = os.environ.copy()
    # Ensure pipelinerl package is importable
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PIPELINERL_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(PIPELINERL_ROOT)
    )

    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
