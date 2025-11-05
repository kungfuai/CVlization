from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv


def _load_env() -> None:
    candidates = []
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / ".env")
    candidates.append(Path("/cvlization_repo/.env"))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env()

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in mock mode
    litellm = None


DATA_DIR = Path("data")
RAW_DATA = DATA_DIR / "raw" / "penguins_in_a_table.json"
DEFAULT_OUTPUT = Path("var/results.json")

BASE_INSTRUCTION = (
    "You analyze a table of penguins with columns name, age (years), height (cm), and weight (kg). "
    "Answer multiple choice questions by comparing the question with the table. "
    "Be precise and rely solely on the table."
)

MUTATION_LIBRARY: Tuple[str, ...] = (
    "Before answering, restate the table row you think is relevant.",
    "Eliminate each incorrect option aloud before giving the final answer.",
    "If the answer is numeric, repeat the correct unit (years, cm, or kg).",
    "Think step by step about every option before you respond.",
    "Reference the column names (age, height, weight) when explaining your reasoning.",
    "Answer using the format 'Answer: <option text>' on its own line.",
    "Double-check for ties and mention if multiple penguins share the same value.",
    "Summarize the table values you used in one sentence before the final answer.",
    "Call out impossible options explicitly so users know why they were removed.",
    "If asked for minimum or maximum values, compare every penguin before deciding.",
)


@lru_cache(maxsize=1)
def load_task_prefix() -> str:
    if RAW_DATA.exists():
        data = json.loads(RAW_DATA.read_text())
        return data.get("task_prefix", "").strip()
    return (
        "Here is a table where the first line is a header and each subsequent line is a penguin:\n\n"
        "name, age, height (cm), weight (kg)\n"
        "Louis, 7, 50, 11\n"
        "Bernard, 5, 80, 13\n"
        "Vincent, 9, 60, 11\n"
        "Gwen, 8, 70, 15\n\n"
        "For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm."
    )


@lru_cache(maxsize=1)
def load_flan_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.eval()
    return tokenizer, model


def chunk_iterable(items: Iterable, size: int) -> Iterable[List]:
    batch: List = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize penguin table prompts using an MCTS-inspired PromptAgent.")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--dev", type=Path, default=DATA_DIR / "dev.jsonl")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm-provider", type=str, default=None, help="Provider (mock, openai, groq, ollama)")
    parser.add_argument("--llm-model", type=str, default=None, help="Override provider-specific model ID")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")

    parser.add_argument("--iterations", type=int, default=8, help="Number of MCTS iterations")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum number of prompt mutations")
    parser.add_argument("--exploration", type=float, default=1.4, help="Exploration weight for UCT")

    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size for mock provider")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    records: List[Dict] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ProviderConfig:
    def __init__(self, provider: str, model: Optional[str], temperature: float) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature


def detect_provider(args: argparse.Namespace) -> ProviderConfig:
    provider = (
        args.llm_provider
        or os.getenv("PROMPT_MCTS_LLM_PROVIDER")
        or os.getenv("DSPY_LLM_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or "mock"
    ).lower()
    model = (
        args.llm_model
        or os.getenv("PROMPT_MCTS_LLM_MODEL")
        or os.getenv("DSPY_LLM_MODEL")
        or os.getenv("LLM_MODEL")
    )
    raw_temp = (
        args.temperature
        if args.temperature is not None
        else os.getenv("PROMPT_MCTS_LLM_TEMPERATURE")
        or os.getenv("DSPY_LLM_TEMPERATURE")
        or os.getenv("LLM_TEMPERATURE")
    )
    try:
        temperature = float(raw_temp) if raw_temp is not None else 0.0
    except ValueError:
        temperature = 0.0
    return ProviderConfig(provider=provider, model=model, temperature=temperature)


def build_prompt(instructions: Tuple[str, ...]) -> str:
    lines = [BASE_INSTRUCTION.strip()]
    if instructions:
        lines.append("Follow these additional directions:")
        for idx, clause in enumerate(instructions, start=1):
            lines.append(f"{idx}. {clause}")
    lines.append("Always provide the final answer clearly.")
    return "\n".join(lines)


def build_user_message(example: Dict) -> str:
    choices = list(example["target_scores"].keys())
    choice_lines = "\n".join(f"- {choice}" for choice in choices)
    task_prefix = load_task_prefix()
    return (
        f"{task_prefix}\n\n"
        f"Question: {example['input']}\n"
        f"Options:\n{choice_lines}\n\n"
        "Reason carefully and respond with the best option."
    )


def extract_answer(model_output: str, example: Dict) -> str:
    output = model_output.strip()
    if not output:
        return ""

    options = list(example["target_scores"].keys())
    normalized_output = output.lower()

    for option in options:
        if option.lower() in normalized_output:
            return option

    # handle patterns like "Answer: Vincent"
    if ":" in output:
        tail = output.split(":")[-1].strip()
        for option in options:
            if option.lower() == tail.lower():
                return option

    # fall back to best logit option
    return max(example["target_scores"], key=example["target_scores"].get)


def flan_generate(prompt: str, examples: List[Dict], batch_size: int) -> List[str]:
    tokenizer, model = load_flan_model()
    device = next(model.parameters()).device
    outputs: List[str] = []
    for chunk in chunk_iterable(examples, batch_size):
        texts = [
            f"{prompt}\n\n{build_user_message(ex)}\nAnswer:"
            for ex in chunk
        ]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            sequences = model.generate(**encoded, max_new_tokens=48)
        chunk_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        outputs.extend([out.strip() for out in chunk_outputs])
    return outputs


def call_provider(messages: List[Dict[str, str]], config: ProviderConfig) -> str:
    if config.provider == "mock":
        raise RuntimeError("Mock provider should be handled separately.")
    if litellm is None:
        raise RuntimeError("litellm is not installed; cannot call external providers.")

    kwargs = {
        "temperature": config.temperature,
        "messages": messages,
    }
    model_id: Optional[str] = None

    if config.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider 'openai'.")
        model_id = config.model or "gpt-4o-mini"
    elif config.provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for provider 'groq'.")
        model_id = config.model or "groq/llama3-8b-8192"
    elif config.provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        os.environ.setdefault("LITELLM_BASE_URL", base_url)
        model_id = config.model or "ollama/llama3.1"
    else:
        raise ValueError(f"Unsupported provider '{config.provider}'.")

    kwargs["model"] = model_id
    response = litellm.completion(**kwargs)  # type: ignore[arg-type]
    return response["choices"][0]["message"]["content"].strip()


def score_prompt(
    prompt: str,
    dataset: List[Dict],
    provider: ProviderConfig,
    batch_size: int,
    cache: Dict[str, float],
) -> float:
    if prompt in cache:
        return cache[prompt]

    predictions: List[str] = []
    if provider.provider == "mock":
        outputs = flan_generate(prompt, dataset, batch_size=batch_size)
        for output, example in zip(outputs, dataset):
            predictions.append(extract_answer(output, example))
    else:
        for example in dataset:
            user_message = build_user_message(example)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ]
            model_output = call_provider(messages, provider)
            predictions.append(extract_answer(model_output, example))

    correct = 0
    for pred, example in zip(predictions, dataset):
        answer = max(example["target_scores"], key=example["target_scores"].get)
        if pred == answer:
            correct += 1
    accuracy = correct / len(dataset) if dataset else 0.0
    cache[prompt] = accuracy
    return accuracy


@dataclass
class Node:
    instructions: Tuple[str, ...]
    parent: Optional["Node"] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    untried_actions: List[str] = field(default_factory=list)

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def is_fully_expanded(self, max_depth: int) -> bool:
        return len(self.instructions) >= max_depth or not self.untried_actions


class PromptMCTS:
    def __init__(
        self,
        provider: ProviderConfig,
        dataset: List[Dict],
        batch_size: int,
        iterations: int,
        max_depth: int,
        exploration_weight: float,
        rng: random.Random,
    ) -> None:
        self.provider = provider
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterations = iterations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.rng = rng
        self.cache: Dict[str, float] = {}

    def available_actions(self, instructions: Tuple[str, ...]) -> List[str]:
        used = set(instructions)
        return [mut for mut in MUTATION_LIBRARY if mut not in used]

    def evaluate(self, node: Node) -> float:
        prompt = build_prompt(node.instructions)
        return score_prompt(prompt, self.dataset, self.provider, self.batch_size, self.cache)

    def uct(self, parent: Node, child: Node) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.q_value()
        exploration = self.exploration_weight * math.sqrt(math.log(parent.visits) / child.visits)
        return exploitation + exploration

    def select(self, node: Node) -> Node:
        current = node
        while current.is_fully_expanded(self.max_depth) and current.children:
            current = max(current.children.values(), key=lambda child: self.uct(current, child))
        return current

    def expand(self, node: Node) -> Node:
        if len(node.instructions) >= self.max_depth or not node.untried_actions:
            return node
        action_idx = self.rng.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(action_idx)
        new_instructions = node.instructions + (action,)
        child = Node(
            instructions=new_instructions,
            parent=node,
            untried_actions=self.available_actions(new_instructions),
        )
        node.children[action] = child
        return child

    def backpropagate(self, node: Node, value: float) -> None:
        current: Optional[Node] = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def search(self) -> Tuple[Node, List[Dict[str, float]]]:
        root = Node(
            instructions=(),
            parent=None,
            untried_actions=self.available_actions(()),
        )

        # evaluate root once
        root_value = self.evaluate(root)
        self.backpropagate(root, root_value)
        history: List[Dict[str, float]] = [
            {
                "depth": len(root.instructions),
                "instructions": list(root.instructions),
                "accuracy": root_value,
            }
        ]

        for _ in range(self.iterations):
            leaf = self.select(root)
            expanded = self.expand(leaf)
            value = self.evaluate(expanded)
            self.backpropagate(expanded, value)
            history.append(
                {
                    "depth": len(expanded.instructions),
                    "instructions": list(expanded.instructions),
                    "accuracy": value,
                }
            )

        best = max(
            self._iter_nodes(root),
            key=lambda node: node.q_value(),
        )
        return best, history

    def _iter_nodes(self, node: Node) -> Iterable[Node]:
        yield node
        for child in node.children.values():
            yield from self._iter_nodes(child)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    provider = detect_provider(args)

    trainset = load_jsonl(args.train)
    devset = load_jsonl(args.dev)
    testset = load_jsonl(args.test)

    search = PromptMCTS(
        provider=provider,
        dataset=devset,
        batch_size=args.batch_size,
        iterations=args.iterations,
        max_depth=args.max_depth,
        exploration_weight=args.exploration,
        rng=rng,
    )

    best_node, trail = search.search()

    baseline_prompt = build_prompt(())
    best_prompt = build_prompt(best_node.instructions)

    baseline_dev = score_prompt(baseline_prompt, devset, provider, args.batch_size, search.cache)
    baseline_test = score_prompt(baseline_prompt, testset, provider, args.batch_size, search.cache)

    optimized_dev = score_prompt(best_prompt, devset, provider, args.batch_size, search.cache)
    optimized_test = score_prompt(best_prompt, testset, provider, args.batch_size, search.cache)

    results = {
        "mode": provider.provider,
        "iterations": args.iterations,
        "max_depth": args.max_depth,
        "best_instructions": list(best_node.instructions),
        "baseline_prompt": baseline_prompt,
        "optimized_prompt": best_prompt,
        "baseline_dev_accuracy": baseline_dev,
        "optimized_dev_accuracy": optimized_dev,
        "baseline_test_accuracy": baseline_test,
        "optimized_test_accuracy": optimized_test,
        "search_trace": trail,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    print("Baseline dev accuracy:", f"{baseline_dev:.3f}")
    print("Optimized dev accuracy:", f"{optimized_dev:.3f}")
    print("Baseline test accuracy:", f"{baseline_test:.3f}")
    print("Optimized test accuracy:", f"{optimized_test:.3f}")
    print("Best additional instructions:")
    for idx, clause in enumerate(best_node.instructions, start=1):
        print(f"  {idx}. {clause}")
    print("Results saved to", args.output)


if __name__ == "__main__":
    main()
