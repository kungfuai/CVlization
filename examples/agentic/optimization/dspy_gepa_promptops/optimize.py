from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import dspy
from dspy import Example

from gepa import api as gepa_api
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

DATA_DIR = Path("data")
DEFAULT_PROMPT = (
    "You are a concise helpdesk assistant for CVlization. Use the shared cache "
    "(~/.cache/cvlization) to avoid repeated downloads and give actionable steps."
)


@dataclass
class ProviderConfig:
    provider: str
    model: str | None
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a helpdesk prompt using DSPy GEPA.")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=DATA_DIR / "dev.jsonl")
    parser.add_argument("--llm-provider", type=str, help="LLM provider (mock, openai, groq, ollama)")
    parser.add_argument("--llm-model", type=str, help="Provider-specific model identifier")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--max-metric-calls", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("var/results.json"))
    return parser.parse_args()


def detect_provider(args: argparse.Namespace) -> ProviderConfig:
    provider = (
        args.llm_provider
        or os.getenv("DSPY_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or "mock"
    ).lower()
    model = args.llm_model or os.getenv("DSPY_LLM_MODEL") or os.getenv("LLM_MODEL")
    temp_raw = (
        args.temperature
        if args.temperature is not None
        else os.getenv("DSPY_LLM_TEMPERATURE")
        or os.getenv("LLM_TEMPERATURE")
    )
    try:
        temperature = float(temp_raw) if temp_raw is not None else 0.2
    except ValueError:
        temperature = 0.2
    return ProviderConfig(provider=provider, model=model, temperature=temperature)


def load_examples(path: Path) -> List[Example]:
    examples: List[Example] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ex = Example(question=record["question"], answer=record["answer"]).with_inputs("question")
            examples.append(ex)
    return examples


def run_mock(trainset: List[Example], devset: List[Example], output_path: Path) -> None:
    baseline_prompt = DEFAULT_PROMPT
    improved_prompt = (
        DEFAULT_PROMPT
        + " Always mention the shared ~/.cache/cvlization directory and remind users to check NVIDIA tooling if GPUs fail."
    )

    def heuristic_score(prompt: str) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        if ".cache/cvlization" in prompt_lower:
            score += 0.6
        if "gpu" in prompt_lower or "nvidia" in prompt_lower:
            score += 0.4
        return min(score, 1.0)

    baseline_score = heuristic_score(baseline_prompt)
    improved_score = heuristic_score(improved_prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "mode": "mock",
                "baseline_prompt": baseline_prompt,
                "optimized_prompt": improved_prompt,
                "baseline_score": baseline_score,
                "optimized_score": improved_score,
            },
            indent=2,
        )
    )
    print("Mock optimization complete. Results written to", output_path)


def run_gepa(trainset: List[Example], devset: List[Example], config: ProviderConfig, args: argparse.Namespace) -> None:
    train_records = [
        {"input": ex.question, "answer": ex.answer, "additional_context": {}}
        for ex in trainset
    ]
    dev_records = [
        {"input": ex.question, "answer": ex.answer, "additional_context": {}}
        for ex in devset
    ]

    model_id = config.model
    if model_id is None:
        if config.provider == "openai":
            model_id = "openai/gpt-4o-mini"
        elif config.provider == "groq":
            model_id = "groq/llama3-8b-8192"
        elif config.provider == "ollama":
            model_id = "ollama/llama3.1"

    adapter = DefaultAdapter(model=model_id)

    result = gepa_api.optimize(
        seed_candidate={"system_prompt": DEFAULT_PROMPT},
        trainset=train_records,
        valset=dev_records,
        adapter=adapter,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(Path("var/gepa_runs")),
        candidate_selection_strategy="pareto",
        reflection_minibatch_size=2,
    )

    best_candidate = result.best_candidate

    baseline_scores = result.history[0].scores if result.history else []
    baseline_score = float(sum(baseline_scores) / len(baseline_scores)) if baseline_scores else 0.0

    adapter_eval = adapter.evaluate(dev_records, best_candidate)
    optimized_score = float(sum(adapter_eval.scores) / len(adapter_eval.scores)) if adapter_eval.scores else 0.0

    output = {
        "mode": "gepa",
        "provider": config.provider,
        "model": model_id,
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "optimized_prompt": best_candidate.get("system_prompt", DEFAULT_PROMPT),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("Optimization complete. Results written to", args.output)


def main() -> None:
    args = parse_args()
    config = detect_provider(args)
    trainset = load_examples(args.train)
    devset = load_examples(args.dev)

    if config.provider == "mock":
        run_mock(trainset, devset, args.output)
    else:
        run_gepa(trainset, devset, config, args)


if __name__ == "__main__":
    main()
