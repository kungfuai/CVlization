"""Prepare Skywork math data for Direct-OPD training.

Downloads a subset of the Skywork-OR1-RL-Data dataset, applies the
DAPO-style prompt template, and saves as JSONL for training.
"""

import argparse
import json
import os

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
    "{question}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def main():
    parser = argparse.ArgumentParser(description="Prepare Skywork math data for Direct-OPD")
    parser.add_argument("--dataset", default="Skywork/Skywork-OR1-RL-Data", help="HuggingFace dataset name")
    parser.add_argument("--split", default="math", help="Dataset split (math or code)")
    parser.add_argument("--max-prompts", type=int, default=500, help="Maximum number of prompts to keep")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    prompts = []
    for i, row in enumerate(ds):
        # Extract the math question from the prompt field
        question = None

        # The dataset stores prompts as chat message lists
        if "prompt" in row and isinstance(row["prompt"], list):
            for msg in row["prompt"]:
                if msg.get("role") == "user":
                    question = msg["content"]
                    break

        if question is None:
            continue

        formatted = PROMPT_TEMPLATE.format(question=question)
        entry = {"prompt": formatted}

        # Preserve ground truth answer if available for evaluation
        if "reward_model" in row and isinstance(row["reward_model"], dict):
            gt = row["reward_model"].get("ground_truth")
            if gt is not None:
                entry["ground_truth"] = gt

        prompts.append(entry)
        if len(prompts) >= args.max_prompts:
            break

    print(f"Collected {len(prompts)} prompts")

    with open(args.output, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
