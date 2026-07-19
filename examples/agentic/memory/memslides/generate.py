#!/usr/bin/env python3
"""Round-0 personalized slide generation driven by long-term user memory.

Demonstrates step (1) of the MemSlides pattern: before any feedback, the deck is
personalized from the user's persistent profile (theme, tone, bullet budget,
avoided topics). Run it twice with different profiles and the same topic to see
the profile — not the prompt — change the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memslides import MemSlidesAgent, MemoryStore, UserPreferences, make_brain


def default_memory_dir() -> str:
    return str(Path.home() / ".cache" / "cvlization" / "memslides")


def load_profile_file(path: Path):
    data = json.loads(path.read_text())
    prefs = UserPreferences.from_dict(data.get("preferences", {}))
    brief = data.get("brief", {})
    user_id = data.get("user_id", "demo-user")
    return user_id, prefs, brief


def main() -> None:
    ap = argparse.ArgumentParser(description="MemSlides round-0 personalized generation")
    ap.add_argument("--profile", type=str, default="data/user_profile.example.json",
                    help="Path to a user profile+brief JSON (seeds long-term memory on first run)")
    ap.add_argument("--user-id", type=str, default=None, help="Override user id from the profile file")
    ap.add_argument("--topic", type=str, default=None, help="Override the deck topic")
    ap.add_argument("--audience", type=str, default=None, help="Override the audience")
    ap.add_argument("--sections", type=str, nargs="*", default=None, help="Override section list")
    ap.add_argument("--provider", type=str, default=None, help="mock | openai | groq | ollama")
    ap.add_argument("--model", type=str, default=None, help="Provider model id")
    ap.add_argument("--memory-dir", type=str, default=default_memory_dir())
    ap.add_argument("--output", type=str, default="artifacts/deck.json")
    args = ap.parse_args()

    prefs = UserPreferences()
    brief = {}
    user_id = args.user_id or "demo-user"
    profile_path = Path(args.profile)
    if profile_path.is_file():
        pu, prefs, brief = load_profile_file(profile_path)
        user_id = args.user_id or pu

    store = MemoryStore(Path(args.memory_dir))
    profile_mem = store.user_profile(user_id)
    # Seed long-term preferences from the profile file on first run only.
    if not profile_mem.path.is_file():
        profile_mem.seed_from(prefs)
        profile_mem.save()
        print(f"[memory] seeded long-term profile for '{user_id}' at {profile_mem.path}", flush=True)
    else:
        print(f"[memory] loaded existing long-term profile for '{user_id}'", flush=True)

    topic = args.topic or brief.get("topic", "Untitled Deck")
    audience = args.audience or brief.get("audience", "general")
    sections = args.sections or brief.get("sections", ["Background", "Approach", "Results"])

    brain = make_brain(args.provider, args.model)
    agent = MemSlidesAgent(brain, store.tool_memory())

    print(f"[generate] brain={brain.name} topic={topic!r} audience={audience!r}", flush=True)
    print(f"[generate] applied preferences: theme={profile_mem.preferences.theme} "
          f"tone={profile_mem.preferences.tone} "
          f"max_bullets={profile_mem.preferences.max_bullets_per_slide}", flush=True)

    deck = agent.generate(topic, audience, sections, profile_mem)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(deck.to_json())
    (out.parent / "deck.txt").write_text(deck.render_text())
    print(f"\n{deck.render_text()}", flush=True)
    print(f"[generate] wrote {out} ({len(deck.slides)} slides)", flush=True)


if __name__ == "__main__":
    main()
