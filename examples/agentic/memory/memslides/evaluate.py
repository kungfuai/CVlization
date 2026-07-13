#!/usr/bin/env python3
"""End-to-end evaluation + smoke test for the MemSlides memory pattern.

Runs three checks and exits non-zero if any fails:

  1. Scoped local revision vs. full regeneration
     -> local must touch fewer slides and preserve more unrelated content.
  2. Host-persistent memory reuse
     -> a second pass reuses tool-memory recipes and the long-term profile
        instead of re-learning (the CPU analogue of "don't re-download weights").
  3. Preference adherence / long-term carry-over
     -> a durable "always ..." instruction is promoted and honored in a brand
        new deck for the same user.

Runs on CPU with the deterministic mock brain by default; pass --provider to
exercise a real LLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from memslides import (
    Deck,
    MemSlidesAgent,
    MemoryStore,
    deck_diff,
    make_brain,
)

FEEDBACK = [
    "Make slide 3 more concise",
    "From now on always use a dark theme",
    "Add a slide about pricing",
    "Make slide 4 more concise",  # repeat of 'shorten' -> should reuse a tool recipe
]

TOPIC = "Adopting Retrieval-Augmented Generation"
AUDIENCE = "enterprise architects"
SECTIONS = ["Motivation", "Architecture", "Evaluation"]


def run_local(memory_dir: Path, user_id: str, session_id: str, brain):
    store = MemoryStore(memory_dir)
    profile = store.user_profile(user_id)
    profile.save()  # ensure seeded/persisted
    working = store.working_memory(session_id)
    tool_mem = store.tool_memory()
    agent = MemSlidesAgent(brain, tool_mem)

    deck = agent.generate(TOPIC, AUDIENCE, SECTIONS, profile)
    touched, preserved = [], []
    for i, fb in enumerate(FEEDBACK, start=1):
        res = agent.revise_turn(deck, fb, i, working, profile)
        deck = res["deck"]
        touched.append(res["surface"]["slides_touched"])
        preserved.append(res["surface"]["preservation_ratio"])
    working.save()
    tool_mem.save()
    profile.save()
    return {
        "deck": deck,
        "avg_touched": sum(touched) / len(touched),
        "avg_preservation": sum(preserved) / len(preserved),
        "tool_hits": tool_mem.hits,
        "tool_misses": tool_mem.misses,
        "profile": profile,
    }


def run_full(memory_dir: Path, user_id: str, brain):
    store = MemoryStore(memory_dir)
    profile = store.user_profile(user_id)
    agent = MemSlidesAgent(brain, store.tool_memory())

    deck = agent.generate(TOPIC, AUDIENCE, SECTIONS, profile)
    touched, preserved, accumulated = [], [], []
    for fb in FEEDBACK:
        accumulated.append(fb)
        before = deck.clone()
        deck = agent.full_regenerate(TOPIC, AUDIENCE, SECTIONS, profile, accumulated)
        s = deck_diff(before, deck)
        touched.append(s["slides_touched"])
        preserved.append(s["preservation_ratio"])
    return {
        "deck": deck,
        "avg_touched": sum(touched) / len(touched),
        "avg_preservation": sum(preserved) / len(preserved),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MemSlides evaluation / smoke test")
    ap.add_argument("--provider", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--memory-dir", type=str,
                    default=str(Path.home() / ".cache" / "cvlization" / "memslides" / "eval"))
    ap.add_argument("--report", type=str, default="artifacts/evaluate_report.json")
    ap.add_argument("--fresh", action="store_true", help="Wipe the eval memory dir before running")
    args = ap.parse_args()

    brain = make_brain(args.provider, args.model)
    root = Path(args.memory_dir)
    if args.fresh and root.exists():
        shutil.rmtree(root)

    print(f"[evaluate] brain={brain.name} memory_dir={root}", flush=True)

    # --- Check 1: scoped local vs full regeneration -------------------------
    local = run_local(root / "compare_local", "cmp-user", "cmp-sess", brain)
    full = run_full(root / "compare_full", "cmp-user", brain)
    check1 = (
        local["avg_touched"] < full["avg_touched"]
        and local["avg_preservation"] > full["avg_preservation"]
    )

    # --- Check 2: host-persistent memory reuse (second pass reuses recipes) --
    reuse_dir = root / "reuse"
    if reuse_dir.exists():
        shutil.rmtree(reuse_dir)
    pass1 = run_local(reuse_dir, "reuse-user", "reuse-sess-1", brain)
    profile_seeded_first = (reuse_dir / "users" / "reuse-user.json").is_file()
    pass2 = run_local(reuse_dir, "reuse-user", "reuse-sess-2", brain)
    # pass1 populates recipes (mostly misses); pass2 should register cache hits.
    check2 = pass2["tool_hits"] > 0 and profile_seeded_first

    # --- Check 3: durable preference promoted & honored in a NEW deck --------
    store = MemoryStore(reuse_dir)
    profile_after = store.user_profile("reuse-user")
    promoted_theme = profile_after.preferences.theme
    fresh_deck = MemSlidesAgent(brain, store.tool_memory()).generate(
        "A Totally Different Deck", "students", ["Intro", "Body", "End"], profile_after
    )
    check3 = promoted_theme == "dark" and fresh_deck.theme == "dark"

    report = {
        "brain": brain.name,
        "check1_scoped_vs_full": {
            "pass": bool(check1),
            "local_avg_slides_touched": round(local["avg_touched"], 3),
            "full_avg_slides_touched": round(full["avg_touched"], 3),
            "local_avg_preservation": round(local["avg_preservation"], 3),
            "full_avg_preservation": round(full["avg_preservation"], 3),
        },
        "check2_memory_reuse": {
            "pass": bool(check2),
            "pass1_tool_hits": pass1["tool_hits"],
            "pass1_tool_misses": pass1["tool_misses"],
            "pass2_tool_hits": pass2["tool_hits"],
            "profile_seeded_after_pass1": profile_seeded_first,
        },
        "check3_preference_adherence": {
            "pass": bool(check3),
            "promoted_theme": promoted_theme,
            "new_deck_theme": fresh_deck.theme,
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- Pretty summary -----------------------------------------------------
    print("\n=== MemSlides evaluation ===", flush=True)
    c1 = report["check1_scoped_vs_full"]
    print(f"[1] scoped-vs-full     : local touches {c1['local_avg_slides_touched']} slides/turn "
          f"(preserve {c1['local_avg_preservation']}) vs full {c1['full_avg_slides_touched']} "
          f"(preserve {c1['full_avg_preservation']})  -> {'PASS' if check1 else 'FAIL'}", flush=True)
    c2 = report["check2_memory_reuse"]
    print(f"[2] memory reuse       : pass1 hits={c2['pass1_tool_hits']} misses={c2['pass1_tool_misses']}, "
          f"pass2 hits={c2['pass2_tool_hits']}  -> {'PASS' if check2 else 'FAIL'}", flush=True)
    c3 = report["check3_preference_adherence"]
    print(f"[3] preference adherence: durable theme '{c3['promoted_theme']}' honored in new deck "
          f"(theme={c3['new_deck_theme']})  -> {'PASS' if check3 else 'FAIL'}", flush=True)

    all_pass = check1 and check2 and check3
    print(f"\n[evaluate] {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}; "
          f"report -> {args.report}", flush=True)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
