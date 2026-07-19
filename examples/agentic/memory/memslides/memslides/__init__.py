"""MemSlides: a minimal hierarchical-memory agent for personalized slide generation.

The transferable contribution demonstrated here is the *tri-partite memory design*
plus *scoped slide-local revision*, not the slide artifact itself:

- UserProfileMemory  (long-term)  -> persistent, cross-deck preferences
- WorkingMemory      (session)    -> constraints + edit-state within one deck
- ToolMemory         (reusable)   -> cached edit recipes across similar operations

See README.md for the architecture overview.
"""

from .deck import Deck, Slide, deck_diff
from .memory import (
    MemoryStore,
    UserPreferences,
    UserProfileMemory,
    WorkingMemory,
    ToolMemory,
)
from .agent import MemSlidesAgent, EditPlan
from .llm import Brain, MockBrain, LLMBrain, make_brain

__all__ = [
    "Deck",
    "Slide",
    "deck_diff",
    "MemoryStore",
    "UserPreferences",
    "UserProfileMemory",
    "WorkingMemory",
    "ToolMemory",
    "MemSlidesAgent",
    "EditPlan",
    "Brain",
    "MockBrain",
    "LLMBrain",
    "make_brain",
]
