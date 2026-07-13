#!/usr/bin/env python3
"""Multi-turn scoped local revision with working + tool + long-term memory.

Demonstrates steps (2) and (3): feedback turns are localized to the smallest
affected slide region and applied as patches. Working memory carries a
preference stated on an early turn into later turns; tool memory reuses edit
recipes; durable feedback ("always ...") is promoted to the long-term profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memslides import Deck, MemSlidesAgent, MemoryStore

from generate import default_memory_dir  # reuse the same default


def load_feedback(args) -> list:
    turns = list(args.feedback or [])
    if args.feedback_file:
        for line in Path(args.feedback_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                turns.append(line)
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description="MemSlides multi-turn scoped local revision")
    ap.add_argument("--deck", type=str, default="artifacts/deck.json", help="Input deck JSON")
    ap.add_argument("--user-id", type=str, default="demo-user")
    ap.add_argument("--session-id", type=str, default="session-1")
    ap.add_argument("--feedback", type=str, nargs="*", default=None, help="One or more feedback turns")
    ap.add_argument("--feedback-file", type=str, default=None, help="Newline-delimited feedback turns")
    ap.add_argument("--mode", choices=["local", "full"], default="local",
                    help="local = scoped patch (default); full = regenerate whole deck each turn")
    ap.add_argument("--provider", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--memory-dir", type=str, default=default_memory_dir())
    ap.add_argument("--output", type=str, default="artifacts/deck_revised.json")
    ap.add_argument("--report", type=str, default="artifacts/revision_report.json")
    args = ap.parse_args()

    from memslides import make_brain

    deck = Deck.from_dict(json.loads(Path(args.deck).read_text()))
    feedback_turns = load_feedback(args)
    if not feedback_turns:
        raise SystemExit("No feedback provided. Use --feedback '...' or --feedback-file path.")

    store = MemoryStore(Path(args.memory_dir))
    profile = store.user_profile(args.user_id)
    working = store.working_memory(args.session_id)
    tool_mem = store.tool_memory()
    brain = make_brain(args.provider, args.model)
    agent = MemSlidesAgent(brain, tool_mem)

    print(f"[revise] brain={brain.name} mode={args.mode} turns={len(feedback_turns)}", flush=True)

    turn_reports = []
    accumulated = []
    for i, fb in enumerate(feedback_turns, start=1):
        accumulated.append(fb)
        if args.mode == "full":
            before = deck.clone()
            deck = agent.full_regenerate(
                deck.topic, deck.audience,
                sections=["Background", "Approach", "Results"],
                profile=profile, accumulated_feedback=accumulated,
            )
            from memslides import deck_diff
            surface = deck_diff(before, deck)
            rep = {"turn": i, "feedback": fb, "mode": "full", "surface": surface,
                   "tool_recipe_reused": False, "promoted_to_profile": None}
        else:
            result = agent.revise_turn(deck, fb, i, working, profile)
            deck = result.pop("deck")
            rep = result
        turn_reports.append(rep)
        s = rep["surface"]
        print(f"  turn {i}: {fb!r}", flush=True)
        print(f"    op={rep['plan']['op'] if 'plan' in rep else 'regenerate'} "
              f"slides_touched={s['slides_touched']} "
              f"preservation={s['preservation_ratio']} "
              f"tool_reused={rep['tool_recipe_reused']}", flush=True)
        if rep.get("promoted_to_profile"):
            print(f"    -> promoted to long-term profile: {rep['promoted_to_profile']}", flush=True)

    # Persist memory state.
    working.save()
    tool_mem.save()
    profile.save()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(deck.to_json())
    (out.parent / "deck_revised.txt").write_text(deck.render_text())

    report = {
        "mode": args.mode,
        "user_id": args.user_id,
        "session_id": args.session_id,
        "brain": brain.name,
        "num_turns": len(feedback_turns),
        "avg_slides_touched": round(sum(r["surface"]["slides_touched"] for r in turn_reports) / len(turn_reports), 3),
        "tool_reuse_count": sum(1 for r in turn_reports if r["tool_recipe_reused"]),
        "working_memory_constraints": working.constraints,
        "profile_after": profile.preferences.to_dict(),
        "turns": turn_reports,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n{deck.render_text()}", flush=True)
    print(f"[revise] wrote {out} and {args.report}", flush=True)
    print(f"[revise] avg_slides_touched={report['avg_slides_touched']} "
          f"tool_reuse_count={report['tool_reuse_count']}", flush=True)


if __name__ == "__main__":
    main()
