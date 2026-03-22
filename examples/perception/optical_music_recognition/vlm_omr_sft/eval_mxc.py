#!/usr/bin/env python3
"""Evaluate MXC transcription accuracy against ground truth.

Computes alignment-aware metrics between predicted and reference MXC sequences.
Reads from WandB inference tables or standalone prediction/reference pairs.

Usage:
    # Evaluate a WandB run's inference tables
    python eval_mxc.py --wandb-run RUN_DIR

    # Evaluate a single prediction file against reference
    python eval_mxc.py --pred pred.mxc --ref ref.mxc

    # Evaluate latest WandB run
    python eval_mxc.py --latest
"""

import argparse
import json
import glob
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class NoteEvent:
    """A single note or rest event extracted from MXC."""
    kind: str        # "N" or "R"
    pitch: str       # e.g. "C4", "Bb3", "R"
    note_type: str   # e.g. "q", "e", "h", "w"
    duration: str    # e.g. "10080"
    raw: str = ""    # full MXC line


@dataclass
class EvalResult:
    """Evaluation metrics for one prediction-reference pair."""
    score_id: str = ""
    step: int = 0

    # Counts
    pred_notes: int = 0       # pitched notes predicted
    ref_notes: int = 0        # pitched notes in reference
    pred_events: int = 0      # all events (notes + rests) predicted
    ref_events: int = 0       # all events in reference

    # Alignment-aware metrics (via SequenceMatcher) — includes rests
    pitch_similarity: float = 0.0      # sequence similarity of pitch tokens (notes + rests)
    rhythm_similarity: float = 0.0     # sequence similarity of type tokens
    combined_similarity: float = 0.0   # sequence similarity of pitch+type pairs

    # Pitched-only metrics — excludes rests, measures only actual note accuracy
    pitched_only_similarity: float = 0.0  # sequence similarity of pitched notes only
    pitched_only_positional: float = 0.0  # positional match of pitched notes only

    # Coverage
    note_coverage: float = 0.0         # pred_events / ref_events

    # Longest matching blocks
    longest_pitch_match: int = 0       # longest consecutive pitch match
    longest_combined_match: int = 0    # longest consecutive pitch+type match

    # Pitch variety
    pred_unique_pitches: int = 0
    ref_unique_pitches: int = 0

    # Header accuracy
    correct_key: bool = False
    correct_time: bool = False
    correct_parts: bool = False


def extract_events(mxc_text: str) -> list[NoteEvent]:
    """Extract note/rest events from MXC text."""
    events = []
    for line in mxc_text.split("\n"):
        line = line.strip()
        # Pitched note: N C4 q 10080 ...
        m = re.match(r"(\+?g?N) (\S+) (\S+) (\d+)", line)
        if m:
            events.append(NoteEvent(
                kind="N", pitch=m.group(2), note_type=m.group(3),
                duration=m.group(4), raw=line,
            ))
            continue
        # Rest: R q 10080 or R whole 30240
        m = re.match(r"(\+?g?R)(?: (\S+))?(?: (\d+))?", line)
        if m and (m.group(2) or m.group(3)):
            events.append(NoteEvent(
                kind="R", pitch="R", note_type=m.group(2) or "?",
                duration=m.group(3) or "0", raw=line,
            ))
    return events


def extract_header(mxc_text: str) -> dict:
    """Extract header info from MXC."""
    header = {}
    for line in mxc_text.split("\n"):
        line = line.strip()
        if line.startswith("M ") and "key=" in line:
            m = re.search(r"key=(\S+)", line)
            if m:
                header["key"] = m.group(1)
        if line.startswith("M ") and "time=" in line:
            m = re.search(r"time=(\S+)", line)
            if m:
                header["time"] = m.group(1)
        if re.match(r"^P\S+\s", line) and not line.startswith("P1 ") or line.startswith("P1 "):
            header.setdefault("parts", []).append(line.split()[0])
    return header


def evaluate_pair(pred_text: str, ref_text: str, score_id: str = "", step: int = 0) -> EvalResult:
    """Evaluate a single prediction against reference."""
    result = EvalResult(score_id=score_id, step=step)

    pred_events = extract_events(pred_text)
    ref_events = extract_events(ref_text)

    result.pred_events = len(pred_events)
    result.ref_events = len(ref_events)
    result.pred_notes = sum(1 for e in pred_events if e.kind == "N")
    result.ref_notes = sum(1 for e in ref_events if e.kind == "N")

    if result.ref_events > 0:
        result.note_coverage = result.pred_events / result.ref_events

    # Pitch variety
    pred_pitches_set = set(e.pitch for e in pred_events if e.kind == "N"
                           and not any(int(d) > 8 for d in re.findall(r"\d+", e.pitch)))
    ref_pitches_set = set(e.pitch for e in ref_events if e.kind == "N")
    result.pred_unique_pitches = len(pred_pitches_set)
    result.ref_unique_pitches = len(ref_pitches_set)

    if not pred_events or not ref_events:
        return result

    # Sequences for comparison
    pred_pitches = [e.pitch for e in pred_events]
    ref_pitches = [e.pitch for e in ref_events]
    pred_types = [e.note_type for e in pred_events]
    ref_types = [e.note_type for e in ref_events]
    pred_combined = [f"{e.pitch}:{e.note_type}" for e in pred_events]
    ref_combined = [f"{e.pitch}:{e.note_type}" for e in ref_events]

    # Pitch similarity (all events including rests)
    sm_pitch = SequenceMatcher(None, pred_pitches, ref_pitches)
    result.pitch_similarity = sm_pitch.ratio()
    blocks = sm_pitch.get_matching_blocks()
    result.longest_pitch_match = max((b.size for b in blocks), default=0)

    # Pitched-only similarity (excludes rests — measures actual note accuracy)
    pred_pitched = [e.pitch for e in pred_events if e.kind == "N"]
    ref_pitched = [e.pitch for e in ref_events if e.kind == "N"]
    if pred_pitched and ref_pitched:
        sm_pitched = SequenceMatcher(None, pred_pitched, ref_pitched)
        result.pitched_only_similarity = sm_pitched.ratio()
        n = min(len(pred_pitched), len(ref_pitched))
        result.pitched_only_positional = (
            sum(1 for p, r in zip(pred_pitched[:n], ref_pitched[:n]) if p == r) / n
        )

    # Rhythm similarity
    sm_type = SequenceMatcher(None, pred_types, ref_types)
    result.rhythm_similarity = sm_type.ratio()

    # Combined similarity
    sm_combined = SequenceMatcher(None, pred_combined, ref_combined)
    result.combined_similarity = sm_combined.ratio()
    blocks_c = sm_combined.get_matching_blocks()
    result.longest_combined_match = max((b.size for b in blocks_c), default=0)

    # Header accuracy
    pred_header = extract_header(pred_text)
    ref_header = extract_header(ref_text)
    result.correct_key = pred_header.get("key") == ref_header.get("key")
    result.correct_time = pred_header.get("time") == ref_header.get("time")
    result.correct_parts = pred_header.get("parts") == ref_header.get("parts")

    return result


def format_result(r: EvalResult) -> str:
    """Format a single result as a readable string."""
    lines = [f"  {r.score_id} (step {r.step}):"]
    lines.append(f"    Events: {r.pred_events}/{r.ref_events} (coverage {100*r.note_coverage:.0f}%)")
    lines.append(f"    Pitched notes: {r.pred_notes}/{r.ref_notes}, unique: {r.pred_unique_pitches}/{r.ref_unique_pitches}")
    lines.append(f"    Pitch similarity:    {100*r.pitch_similarity:.0f}% (longest match: {r.longest_pitch_match})")
    lines.append(f"    Pitched-only sim:    {100*r.pitched_only_similarity:.0f}% (positional: {100*r.pitched_only_positional:.0f}%)")
    lines.append(f"    Rhythm similarity:   {100*r.rhythm_similarity:.0f}%")
    lines.append(f"    Combined similarity: {100*r.combined_similarity:.0f}% (longest match: {r.longest_combined_match})")
    lines.append(f"    Header: key={'✓' if r.correct_key else '✗'} time={'✓' if r.correct_time else '✗'} parts={'✓' if r.correct_parts else '✗'}")
    return "\n".join(lines)


def format_summary(results: list[EvalResult]) -> str:
    """Format aggregate summary across multiple results."""
    if not results:
        return "No results."

    n = len(results)
    avg = lambda vals: sum(vals) / len(vals) if vals else 0

    lines = [f"Aggregate ({n} samples):"]
    lines.append(f"  Avg pitch similarity:    {100*avg([r.pitch_similarity for r in results]):.0f}%")
    lines.append(f"  Avg pitched-only sim:    {100*avg([r.pitched_only_similarity for r in results]):.0f}% (positional: {100*avg([r.pitched_only_positional for r in results]):.0f}%)")
    lines.append(f"  Avg rhythm similarity:   {100*avg([r.rhythm_similarity for r in results]):.0f}%")
    lines.append(f"  Avg combined similarity: {100*avg([r.combined_similarity for r in results]):.0f}%")
    lines.append(f"  Avg note coverage:       {100*avg([r.note_coverage for r in results]):.0f}%")
    lines.append(f"  Avg unique pitches:      {avg([r.pred_unique_pitches for r in results]):.1f} pred / {avg([r.ref_unique_pitches for r in results]):.1f} ref")
    lines.append(f"  Max pitch similarity:    {100*max(r.pitch_similarity for r in results):.0f}%")
    lines.append(f"  Max longest match:       {max(r.longest_pitch_match for r in results)} events")
    lines.append(f"  Header accuracy:         key={sum(r.correct_key for r in results)}/{n} time={sum(r.correct_time for r in results)}/{n} parts={sum(r.correct_parts for r in results)}/{n}")
    return "\n".join(lines)


def eval_wandb_tables(run_dir: str, step: int = None) -> list[EvalResult]:
    """Evaluate all inference tables from a WandB run directory."""
    tables = sorted(glob.glob(f"{run_dir}/files/media/table/*.table.json"))
    if not tables:
        print(f"No tables found in {run_dir}")
        return []

    results = []
    for t in tables:
        with open(t) as f:
            data = json.load(f)
        step_col = data["columns"].index("step")
        pred_col = data["columns"].index("prediction")
        ref_col = data["columns"].index("reference")
        sid_col = data["columns"].index("score_id")

        for row in data["data"]:
            s = row[step_col]
            if step is not None and s != step:
                continue
            if s == 0:
                continue
            result = evaluate_pair(
                row[pred_col], row[ref_col],
                score_id=row[sid_col], step=s,
            )
            results.append(result)

    return results


def find_latest_run() -> str:
    """Find the most recent WandB run directory."""
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        wandb_dir = Path("examples/perception/optical_music_recognition/vlm_omr_sft/wandb")
    runs = sorted(wandb_dir.glob("run-*"))
    if not runs:
        print("No WandB runs found")
        sys.exit(1)
    return str(runs[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb-run", help="WandB run directory")
    parser.add_argument("--latest", action="store_true", help="Use latest WandB run")
    parser.add_argument("--step", type=int, default=None, help="Only evaluate this step")
    parser.add_argument("--pred", help="Prediction MXC file")
    parser.add_argument("--ref", help="Reference MXC file")
    parser.add_argument("--by-step", action="store_true", help="Show summary per step")
    args = parser.parse_args()

    if args.pred and args.ref:
        pred_text = Path(args.pred).read_text()
        ref_text = Path(args.ref).read_text()
        result = evaluate_pair(pred_text, ref_text)
        print(format_result(result))
        return

    run_dir = args.wandb_run
    if args.latest:
        run_dir = find_latest_run()
    if not run_dir:
        parser.print_help()
        return

    print(f"Evaluating: {run_dir}")
    results = eval_wandb_tables(run_dir, step=args.step)

    if not results:
        print("No results found.")
        return

    if args.by_step:
        by_step = defaultdict(list)
        for r in results:
            by_step[r.step].append(r)
        for step in sorted(by_step.keys()):
            step_results = by_step[step]
            print(f"\n--- Step {step} ---")
            print(format_summary(step_results))
    else:
        # Show per-sample results for the latest step
        latest_step = max(r.step for r in results)
        latest = [r for r in results if r.step == latest_step]
        print(f"\n--- Step {latest_step} (latest) ---")
        for r in latest:
            print(format_result(r))
        print()
        print(format_summary(latest))

        # Also show trend if multiple steps
        steps = sorted(set(r.step for r in results))
        if len(steps) > 1:
            print(f"\n--- Trend across {len(steps)} steps ---")
            print(f"{'Step':>6} | {'Pitch':>6} | {'Notes':>6} | {'Rhythm':>6} | {'Combined':>8} | {'Coverage':>8} | {'Unique':>6}")
            print("-" * 70)
            for step in steps:
                sr = [r for r in results if r.step == step]
                avg = lambda vals: sum(vals) / len(vals) if vals else 0
                print(f"{step:6d} | {100*avg([r.pitch_similarity for r in sr]):5.0f}% | {100*avg([r.pitched_only_similarity for r in sr]):5.0f}% | {100*avg([r.rhythm_similarity for r in sr]):5.0f}% | {100*avg([r.combined_similarity for r in sr]):7.0f}% | {100*avg([r.note_coverage for r in sr]):7.0f}% | {avg([r.pred_unique_pitches for r in sr]):5.1f}")


if __name__ == "__main__":
    main()
