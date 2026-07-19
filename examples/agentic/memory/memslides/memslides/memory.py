"""Hierarchical memory: the transferable core of MemSlides.

Three memory layers, each with a different lifetime and scope:

    UserProfileMemory  (long-term)  persists across sessions/decks, keyed by user.
    WorkingMemory      (session)    lives for one deck-authoring session.
    ToolMemory         (reusable)   caches successful edit recipes across sessions.

All three persist as small JSON files under a memory directory so the pattern is
inspectable and the demo is reproducible on CPU with no external services.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# User profile (long-term)                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class UserPreferences:
    """Durable, cross-deck preferences learned about a user."""

    theme: str = "light"
    tone: str = "professional"
    max_bullets_per_slide: int = 5
    preferred_layouts: List[str] = field(default_factory=lambda: ["title-bullets"])
    avoid_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UserPreferences":
        base = UserPreferences()
        return UserPreferences(
            theme=d.get("theme", base.theme),
            tone=d.get("tone", base.tone),
            max_bullets_per_slide=int(d.get("max_bullets_per_slide", base.max_bullets_per_slide)),
            preferred_layouts=list(d.get("preferred_layouts", base.preferred_layouts)),
            avoid_topics=list(d.get("avoid_topics", base.avoid_topics)),
        )


class UserProfileMemory:
    """Long-term preferences + a rolling log of learned facts, per user."""

    def __init__(self, user_id: str, path: Path):
        self.user_id = user_id
        self.path = path
        self.preferences = UserPreferences()
        self.learned: List[Dict[str, Any]] = []  # audit trail of promoted prefs
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            data = json.loads(self.path.read_text())
            self.preferences = UserPreferences.from_dict(data.get("preferences", {}))
            self.learned = list(data.get("learned", []))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "user_id": self.user_id,
                    "preferences": self.preferences.to_dict(),
                    "learned": self.learned,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def seed_from(self, prefs: UserPreferences) -> None:
        """Seed preferences from an explicit profile file (first run only)."""
        self.preferences = prefs

    def promote(self, field_name: str, value: Any, evidence: str) -> None:
        """Write a durable preference into long-term memory with provenance."""
        if not hasattr(self.preferences, field_name):
            return
        setattr(self.preferences, field_name, value)
        self.learned.append({"field": field_name, "value": value, "evidence": evidence})


# --------------------------------------------------------------------------- #
# Working memory (session)                                                     #
# --------------------------------------------------------------------------- #
class WorkingMemory:
    """Session-scoped constraints + edit-state across feedback turns.

    This is what carries a preference expressed on turn 1 forward to turn 3,
    even before (or without) promoting it to the long-term profile.
    """

    def __init__(self, session_id: str, path: Optional[Path] = None):
        self.session_id = session_id
        self.path = path
        self.constraints: Dict[str, Any] = {}      # e.g. {"tone": "casual"}
        self.edit_log: List[Dict[str, Any]] = []   # ordered record of applied edits
        if path is not None and path.is_file():
            data = json.loads(path.read_text())
            self.constraints = dict(data.get("constraints", {}))
            self.edit_log = list(data.get("edit_log", []))

    def set_constraint(self, key: str, value: Any) -> None:
        self.constraints[key] = value

    def record_edit(self, turn: int, feedback: str, plan: Dict[str, Any], surface: Dict[str, Any]) -> None:
        self.edit_log.append(
            {"turn": turn, "feedback": feedback, "plan": plan, "surface": surface}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "constraints": self.constraints,
            "edit_log": self.edit_log,
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# Tool memory (reusable execution experience)                                 #
# --------------------------------------------------------------------------- #
class ToolMemory:
    """Cache of edit recipes keyed by operation signature.

    When the agent localizes a feedback turn to an operation it has run before
    (e.g. "shorten"), it reuses the stored recipe instead of re-planning, and
    logs the hit. This is what prevents repeated planning for similar edits.
    """

    def __init__(self, path: Path):
        self.path = path
        self.recipes: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if path.is_file():
            data = json.loads(path.read_text())
            self.recipes = dict(data.get("recipes", {}))

    def retrieve(self, signature: str) -> Optional[Dict[str, Any]]:
        recipe = self.recipes.get(signature)
        if recipe is not None:
            self.hits += 1
            recipe = dict(recipe)
            recipe["uses"] = int(recipe.get("uses", 0)) + 1
            self.recipes[signature] = recipe
        else:
            self.misses += 1
        return recipe

    def store(self, signature: str, recipe: Dict[str, Any]) -> None:
        existing = self.recipes.get(signature, {})
        merged = dict(recipe)
        merged["uses"] = int(existing.get("uses", 0)) + 1
        self.recipes[signature] = merged

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"recipes": self.recipes}, indent=2, ensure_ascii=False)
        )


# --------------------------------------------------------------------------- #
# Store that wires the three layers to a directory                            #
# --------------------------------------------------------------------------- #
class MemoryStore:
    """Filesystem-backed factory for the three memory layers.

    Layout under ``root``::

        users/<user_id>.json     -> UserProfileMemory
        sessions/<session>.json  -> WorkingMemory
        tools.json               -> ToolMemory   (shared across users/sessions)
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def user_profile(self, user_id: str) -> UserProfileMemory:
        return UserProfileMemory(user_id, self.root / "users" / f"{user_id}.json")

    def working_memory(self, session_id: str) -> WorkingMemory:
        return WorkingMemory(session_id, self.root / "sessions" / f"{session_id}.json")

    def tool_memory(self) -> ToolMemory:
        return ToolMemory(self.root / "tools.json")
