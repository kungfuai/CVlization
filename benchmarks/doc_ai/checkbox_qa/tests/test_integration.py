#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
BENCHMARK_DIR = ROOT / "benchmarks" / "doc_ai" / "checkbox_qa"

SINGLE_DOC_JSON = BENCHMARK_DIR / "data" / "temp_single.jsonl"

def run_command(cmd, env=None):
    result = subprocess.run(cmd, cwd=BENCHMARK_DIR, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def test_prevailing_wage_pipeline():
    """
    Integration test:
      1. Ensure we have a single-doc subset (first doc from gold).
      2. Start vLLM server via run_with_vllm.sh (Qwen 2B, single page).
      3. Run benchmark with the prevailing wage question.
      4. Assert predicted value matches expected ("OES").
    """
    assert SINGLE_DOC_JSON.exists(), "temp_single.jsonl not found; create it first."

    env = os.environ.copy()
    env["QWEN3_VL_MAX_PAGES"] = "1"

    # Run benchmark for the single-doc subset
    run_command(["./run_with_vllm.sh", "qwen3_vl_2b", "--subset", str(SINGLE_DOC_JSON)], env=env)

    latest = sorted((BENCHMARK_DIR / "results").iterdir())[-1]
    pred_path = latest / "qwen3_vl_2b" / "predictions.jsonl"
    assert pred_path.exists(), f"Predictions not found at {pred_path}"

    with open(pred_path) as f:
        doc = json.loads(f.readline())

    answers = {ann["key"]: ann["values"][0]["value"] for ann in doc["annotations"]}
    assert answers["What is the prevailing wage source?"].strip('"') == "OES"
