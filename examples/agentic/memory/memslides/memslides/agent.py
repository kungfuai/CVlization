"""MemSlidesAgent: orchestrates the three memory layers with scoped revision.

The agent is deliberately thin. All *reasoning* lives in a :class:`Brain`
(mock or LLM); all *memory* lives in the layers from ``memory.py``. The agent
wires them together and applies mechanical, minimal patches so that a one-slide
request touches one slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .deck import Deck, Slide, deck_diff
from .llm import Brain
from .memory import ToolMemory, UserProfileMemory, WorkingMemory


@dataclass
class EditPlan:
    op: str
    scope: str
    params: Dict[str, Any] = field(default_factory=dict)
    durable: bool = False
    preference: Optional[List[Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EditPlan":
        return EditPlan(
            op=d.get("op", "add_bullet"),
            scope=d.get("scope", "slide"),
            params=dict(d.get("params", {})),
            durable=bool(d.get("durable", False)),
            preference=list(d["preference"]) if d.get("preference") else None,
        )

    def signature(self) -> str:
        """Operation signature used as the ToolMemory key (params-agnostic)."""
        return f"{self.scope}:{self.op}"


class MemSlidesAgent:
    def __init__(self, brain: Brain, tool_memory: ToolMemory):
        self.brain = brain
        self.tool_memory = tool_memory

    # ------------------------------------------------------------------ #
    # Round-0 personalization                                            #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        topic: str,
        audience: str,
        sections: List[str],
        profile: UserProfileMemory,
    ) -> Deck:
        """Personalized generation before any feedback, driven by long-term memory."""
        return self.brain.generate_deck(topic, audience, sections, profile.preferences)

    # ------------------------------------------------------------------ #
    # Scoped local revision                                              #
    # ------------------------------------------------------------------ #
    def revise_turn(
        self,
        deck: Deck,
        feedback: str,
        turn: int,
        working: WorkingMemory,
        profile: UserProfileMemory,
    ) -> Dict[str, Any]:
        """Apply one feedback turn as a scoped local edit.

        Returns a per-turn report including the edit plan, whether a cached
        tool recipe was reused, the measured edit surface, and any promotion
        to long-term memory.
        """
        plan = EditPlan.from_dict(self.brain.localize(feedback, deck))

        # Tool memory: reuse a recipe for this operation if we've done it before.
        sig = plan.signature()
        cached = self.tool_memory.retrieve(sig)
        reused = cached is not None
        recipe = {"op": plan.op, "scope": plan.scope, "param_keys": sorted(plan.params.keys())}
        self.tool_memory.store(sig, recipe)

        before = deck.clone()
        new_deck = self._apply(deck, plan)
        surface = deck_diff(before, new_deck)

        # Working memory: carry constraint + edit-state to later turns.
        if plan.op in ("set_tone", "set_theme"):
            working.set_constraint(plan.op.replace("set_", ""), list(plan.params.values())[0])
        working.record_edit(turn, feedback, self._plan_dict(plan), surface)

        # Long-term promotion when the user implies a durable preference.
        promoted = None
        if plan.durable and plan.preference:
            field_name, value = plan.preference
            profile.promote(field_name, value, evidence=feedback.strip())
            promoted = {"field": field_name, "value": value}

        return {
            "turn": turn,
            "feedback": feedback,
            "plan": self._plan_dict(plan),
            "tool_recipe_reused": reused,
            "surface": surface,
            "promoted_to_profile": promoted,
            "deck": new_deck,
        }

    # ------------------------------------------------------------------ #
    # Baseline: full regeneration (for the comparison in evaluate)        #
    # ------------------------------------------------------------------ #
    def full_regenerate(
        self,
        topic: str,
        audience: str,
        sections: List[str],
        profile: UserProfileMemory,
        accumulated_feedback: List[str],
    ) -> Deck:
        """Rebuild the whole deck from scratch, folding feedback into sections.

        This is the naive alternative to scoped revision: it re-emits every
        slide, so unrelated content churns even when only one slide was targeted.
        """
        extra = [f"(feedback) {fb}" for fb in accumulated_feedback]
        return self.brain.generate_deck(topic, audience, sections + extra, profile.preferences)

    # ------------------------------------------------------------------ #
    # Mechanical patch application (the "smallest affected region")       #
    # ------------------------------------------------------------------ #
    def _apply(self, deck: Deck, plan: EditPlan) -> Deck:
        d = deck.clone()
        p = plan.params
        op = plan.op

        if op == "set_theme":
            d.theme = p.get("theme", d.theme)
        elif op == "set_tone":
            d.tone = p.get("tone", d.tone)
        elif op == "shorten":
            sid = p.get("slide_id")
            n = int(p.get("max_bullets", 3))
            if sid is not None:
                s = d.slide_by_id(int(sid))
                s.bullets = s.bullets[: max(1, n)]
        elif op == "retitle":
            sid = p.get("slide_id")
            if sid is not None:
                d.slide_by_id(int(sid)).title = p.get("title", d.slide_by_id(int(sid)).title)
        elif op == "add_bullet":
            sid = p.get("slide_id")
            if sid is not None and p.get("bullet"):
                d.slide_by_id(int(sid)).bullets.append(p["bullet"])
        elif op == "remove_slide":
            sid = p.get("slide_id")
            if sid is not None:
                d.slides = [s for s in d.slides if s.id != int(sid)]
        elif op == "add_slide":
            new = Slide(
                id=d.next_slide_id(),
                title=p.get("title", "New Slide"),
                bullets=list(p.get("bullets", [f"Key point about {p.get('title', 'the topic').lower()}"])),
                layout=p.get("layout", "title-bullets"),
            )
            after = p.get("after_id")
            if after is not None and any(s.id == int(after) for s in d.slides):
                idx = next(i for i, s in enumerate(d.slides) if s.id == int(after))
                d.slides.insert(idx + 1, new)
            else:
                d.slides.append(new)
        # unknown op -> no-op (deck unchanged)
        return d

    @staticmethod
    def _plan_dict(plan: EditPlan) -> Dict[str, Any]:
        return {
            "op": plan.op,
            "scope": plan.scope,
            "params": plan.params,
            "durable": plan.durable,
            "preference": plan.preference,
        }
