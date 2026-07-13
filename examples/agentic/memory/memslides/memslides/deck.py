"""Deck / Slide data model and diff metrics.

A deck is a plain, JSON-serializable structure so the whole pipeline stays
CPU-only and dependency-light. The diff helpers make "edit surface" measurable,
which is what lets us quantify scoped local revision vs. full regeneration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Slide:
    id: int
    title: str
    bullets: List[str] = field(default_factory=list)
    layout: str = "title-bullets"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Slide":
        return Slide(
            id=int(d["id"]),
            title=d.get("title", ""),
            bullets=list(d.get("bullets", [])),
            layout=d.get("layout", "title-bullets"),
            notes=d.get("notes", ""),
        )

    def fingerprint(self) -> str:
        """Stable content signature, ignoring the slide id/order."""
        return json.dumps(
            {"title": self.title, "bullets": self.bullets, "layout": self.layout, "notes": self.notes},
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass
class Deck:
    topic: str
    theme: str = "light"
    audience: str = "general"
    tone: str = "professional"
    slides: List[Slide] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "theme": self.theme,
            "audience": self.audience,
            "tone": self.tone,
            "slides": [s.to_dict() for s in self.slides],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Deck":
        return Deck(
            topic=d.get("topic", ""),
            theme=d.get("theme", "light"),
            audience=d.get("audience", "general"),
            tone=d.get("tone", "professional"),
            slides=[Slide.from_dict(s) for s in d.get("slides", [])],
        )

    def clone(self) -> "Deck":
        return Deck.from_dict(json.loads(json.dumps(self.to_dict())))

    def slide_by_id(self, slide_id: int) -> Slide:
        for s in self.slides:
            if s.id == slide_id:
                return s
        raise KeyError(f"No slide with id={slide_id}")

    def next_slide_id(self) -> int:
        return (max((s.id for s in self.slides), default=0)) + 1

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def render_text(self) -> str:
        """Human-readable rendering used for eyeballing outputs."""
        lines = [
            f"# Deck: {self.topic}",
            f"theme={self.theme} | audience={self.audience} | tone={self.tone} | slides={len(self.slides)}",
            "",
        ]
        for s in self.slides:
            lines.append(f"[{s.id}] ({s.layout}) {s.title}")
            for b in s.bullets:
                lines.append(f"    - {b}")
            if s.notes:
                lines.append(f"    (notes: {s.notes})")
            lines.append("")
        return "\n".join(lines)


def deck_diff(old: Deck, new: Deck) -> Dict[str, Any]:
    """Measure the edit surface between two decks.

    Slides are matched by id. A matched slide counts as "changed" when its
    content fingerprint differs. Deck-level fields (theme/tone/audience) are
    reported separately so a pure theme swap is not miscounted as slide churn.
    """
    old_by_id = {s.id: s for s in old.slides}
    new_by_id = {s.id: s for s in new.slides}

    changed_ids: List[int] = []
    unchanged = 0
    for sid, ns in new_by_id.items():
        if sid in old_by_id:
            if old_by_id[sid].fingerprint() != ns.fingerprint():
                changed_ids.append(sid)
            else:
                unchanged += 1

    added_ids = sorted(set(new_by_id) - set(old_by_id))
    removed_ids = sorted(set(old_by_id) - set(new_by_id))

    deck_fields_changed = [
        f
        for f in ("theme", "tone", "audience")
        if getattr(old, f) != getattr(new, f)
    ]

    total_old = max(len(old.slides), 1)
    touched = len(changed_ids) + len(added_ids) + len(removed_ids)
    return {
        "changed_slide_ids": sorted(changed_ids),
        "added_slide_ids": added_ids,
        "removed_slide_ids": removed_ids,
        "unchanged_slides": unchanged,
        "deck_fields_changed": deck_fields_changed,
        "slides_touched": touched,
        # fraction of the *previous* deck that survived byte-identical
        "preservation_ratio": round(unchanged / total_old, 3),
    }
