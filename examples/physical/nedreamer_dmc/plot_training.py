#!/usr/bin/env python3
"""Plot training curve from NE-Dreamer metrics.jsonl logs."""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="outputs/logdir")
    parser.add_argument("--output", type=str, default="outputs/training_curve.png")
    parser.add_argument("--metric", type=str, default="episode/score",
                        help="Metric to plot (default: episode/score)")
    args = parser.parse_args()

    metrics_file = Path(args.logdir) / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"No metrics found at {metrics_file}")
        return

    steps, values = [], []
    eval_steps, eval_values = [], []
    with open(metrics_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry.get("step", 0)
            if args.metric in entry:
                steps.append(step)
                values.append(entry[args.metric])
            if "episode/eval_score" in entry:
                eval_steps.append(step)
                eval_values.append(entry["episode/eval_score"])

    if not steps and not eval_steps:
        print(f"No data found for metric '{args.metric}'")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    if steps:
        ax.plot(steps, values, alpha=0.3, color="steelblue", label="train episode score")
        # Rolling average
        window = max(1, len(values) // 20)
        if len(values) > window:
            smoothed = [sum(values[max(0,i-window):i+1]) / len(values[max(0,i-window):i+1])
                        for i in range(len(values))]
            ax.plot(steps, smoothed, color="steelblue", linewidth=2, label=f"train (smoothed)")

    if eval_steps:
        ax.plot(eval_steps, eval_values, "o-", color="darkorange", markersize=4,
                linewidth=2, label="eval score")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Score")
    ax.set_title("NE-Dreamer Training on DMC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved training curve to {out}")


if __name__ == "__main__":
    main()
