"""Reasoning backends ("brains") for the MemSlides agent.

Two interchangeable implementations behind one interface:

- ``MockBrain``  deterministic, rule-based, zero-dependency. Used for CI /
  verification and for anyone without an API key. It is a genuine functional
  stand-in, not a stub: it generates topic-aware decks and localizes feedback.
- ``LLMBrain``   prompts a real model through LiteLLM (OpenAI / Groq / Ollama)
  and parses JSON responses. Same interface, higher quality.

The agent (``agent.py``) depends only on the ``Brain`` interface, so swapping
backends changes quality, not orchestration.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .deck import Deck, Slide
from .memory import UserPreferences

try:  # optional dependency: only needed for real providers
    import litellm  # type: ignore
except Exception:  # pragma: no cover - litellm absent in pure-mock installs
    litellm = None


# --------------------------------------------------------------------------- #
# Interface                                                                    #
# --------------------------------------------------------------------------- #
class Brain:
    """Reasoning interface used by :class:`MemSlidesAgent`."""

    name = "brain"

    def generate_deck(
        self, topic: str, audience: str, sections: List[str], prefs: UserPreferences
    ) -> Deck:
        raise NotImplementedError

    def localize(self, feedback: str, deck: Deck) -> Dict[str, Any]:
        """Map a feedback utterance to a scoped edit plan (see agent.EditPlan)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic backend                                                        #
# --------------------------------------------------------------------------- #
_TONE_PREFIX = {
    "professional": "",
    "technical": "Technical detail: ",
    "casual": "Quick take: ",
    "executive": "Bottom line: ",
}


class MockBrain(Brain):
    """Rule-based, deterministic reasoning. No network, no randomness."""

    name = "mock"

    # -- generation -------------------------------------------------------- #
    def generate_deck(
        self, topic: str, audience: str, sections: List[str], prefs: UserPreferences
    ) -> Deck:
        sections = [s for s in sections if s.lower() not in {a.lower() for a in prefs.avoid_topics}]
        if not sections:
            sections = ["Background", "Approach", "Results"]

        layout = prefs.preferred_layouts[0] if prefs.preferred_layouts else "title-bullets"
        prefix = _TONE_PREFIX.get(prefs.tone, "")
        slides: List[Slide] = [
            Slide(id=1, title=topic, bullets=[f"Prepared for: {audience}"], layout="section"),
            Slide(
                id=2,
                title="Agenda",
                bullets=self._cap([f"{prefix}{s}" for s in sections], prefs.max_bullets_per_slide),
                layout=layout,
            ),
        ]
        sid = 3
        for section in sections:
            bullets = [
                f"{prefix}Why {section.lower()} matters for {audience}",
                f"{prefix}Key point about {section.lower()}",
                f"{prefix}Concrete example of {section.lower()}",
                f"{prefix}Implication for {audience}",
            ]
            slides.append(
                Slide(
                    id=sid,
                    title=section,
                    bullets=self._cap(bullets, prefs.max_bullets_per_slide),
                    layout=layout,
                )
            )
            sid += 1
        slides.append(
            Slide(
                id=sid,
                title="Summary",
                bullets=self._cap(
                    [f"{prefix}Recap: {s}" for s in sections] + ["Next steps"],
                    prefs.max_bullets_per_slide,
                ),
                layout=layout,
            )
        )
        return Deck(
            topic=topic,
            theme=prefs.theme,
            audience=audience,
            tone=prefs.tone,
            slides=slides,
        )

    @staticmethod
    def _cap(bullets: List[str], max_bullets: int) -> List[str]:
        return bullets[: max(1, int(max_bullets))]

    # -- localization ------------------------------------------------------ #
    def localize(self, feedback: str, deck: Deck) -> Dict[str, Any]:
        text = feedback.lower().strip()
        durable = any(
            marker in text
            for marker in ("always", "from now on", "in general", "every deck", "going forward")
        )
        target = self._resolve_target(text, deck)

        # deck-level: theme
        m = re.search(r"\b(dark|light|minimal|high[- ]contrast)\b", text)
        if "theme" in text or ("mode" in text and m):
            theme = (m.group(1) if m else "dark").replace(" ", "-")
            return self._plan("set_theme", scope="deck", params={"theme": theme},
                              durable=durable, preference=("theme", theme))

        # deck-level: tone
        m = re.search(r"\b(casual|professional|technical|executive)\b", text)
        if m and ("tone" in text or "voice" in text or "style" in text or durable):
            tone = m.group(1)
            return self._plan("set_tone", scope="deck", params={"tone": tone},
                              durable=durable, preference=("tone", tone))

        # add a slide
        if "add" in text and "slide" in text:
            about = self._extract_topic(text)
            return self._plan("add_slide", scope="slide",
                              params={"title": about.title(), "after_id": target})

        # remove a slide
        if ("remove" in text or "delete" in text or "drop" in text) and target is not None:
            return self._plan("remove_slide", scope="slide", params={"slide_id": target})

        # shorten / trim
        if any(w in text for w in ("concise", "shorter", "trim", "fewer bullet", "too long", "tighten")):
            new_max = 3
            m2 = re.search(r"(\d+)\s+bullet", text)
            if m2:
                new_max = int(m2.group(1))
            pref = ("max_bullets_per_slide", new_max) if durable else None
            return self._plan("shorten", scope="slide",
                              params={"slide_id": target, "max_bullets": new_max},
                              durable=durable, preference=pref)

        # retitle
        if "rename" in text or "title" in text:
            new_title = self._extract_topic(text).title()
            return self._plan("retitle", scope="slide",
                              params={"slide_id": target, "title": new_title})

        # fallback: append a clarifying bullet to the best-matching slide
        return self._plan("add_bullet", scope="slide",
                          params={"slide_id": target, "bullet": feedback.strip()})

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _plan(op, scope, params, durable=False, preference=None) -> Dict[str, Any]:
        return {"op": op, "scope": scope, "params": params, "durable": durable,
                "preference": list(preference) if preference else None}

    @staticmethod
    def _resolve_target(text: str, deck: Deck) -> Optional[int]:
        m = re.search(r"slide\s+#?(\d+)", text)
        if m:
            sid = int(m.group(1))
            if any(s.id == sid for s in deck.slides):
                return sid
        # keyword match against titles/bullets
        best, best_score = None, 0
        words = set(re.findall(r"[a-z]{4,}", text))
        for s in deck.slides:
            hay = (s.title + " " + " ".join(s.bullets)).lower()
            score = sum(1 for w in words if w in hay)
            if score > best_score:
                best, best_score = s.id, score
        if best is not None:
            return best
        return deck.slides[-1].id if deck.slides else None

    @staticmethod
    def _extract_topic(text: str) -> str:
        m = re.search(r"about\s+(.+)$", text)
        if m:
            return re.sub(r"[.!?]+$", "", m.group(1)).strip()
        m = re.search(r"(?:to|as)\s+(.+)$", text)
        if m:
            return re.sub(r"[.!?]+$", "", m.group(1)).strip()
        return "New Topic"


# --------------------------------------------------------------------------- #
# LiteLLM backend                                                              #
# --------------------------------------------------------------------------- #
_GEN_SCHEMA = """Return ONLY JSON:
{"slides":[{"title": str, "bullets": [str, ...], "layout": "title-bullets|section"}]}"""

_LOCALIZE_SCHEMA = """Return ONLY JSON describing the SMALLEST scoped edit:
{"op":"set_theme|set_tone|add_slide|remove_slide|shorten|retitle|add_bullet",
 "scope":"deck|slide",
 "params":{...},          // op-specific: theme/tone/slide_id/max_bullets/title/bullet/after_id
 "durable": bool,          // true only if the user implies a lasting cross-deck preference
 "preference": ["field","value"] or null}  // profile field to promote, if durable"""


class LLMBrain(Brain):
    """Real-provider backend via LiteLLM (OpenAI / Groq / Ollama / etc.)."""

    def __init__(self, provider: str, model: str, temperature: float = 0.2):
        if litellm is None:
            raise RuntimeError("litellm is not installed; use provider=mock or `pip install litellm`.")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.name = f"{provider}:{model}"
        self._configure()

    def _configure(self) -> None:
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for provider 'openai'.")
            self._model_id = self.model
        elif self.provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY is required for provider 'groq'.")
            self._model_id = f"groq/{self.model}"
        elif self.provider == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            os.environ.setdefault("OLLAMA_API_BASE", base)
            self._model_id = f"ollama/{self.model}"
        else:
            # let litellm route arbitrary "provider/model" ids
            self._model_id = self.model

    def _chat(self, system: str, user: str) -> Dict[str, Any]:
        resp = litellm.completion(
            model=self._model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        content = resp["choices"][0]["message"]["content"]
        return _extract_json(content)

    def generate_deck(
        self, topic: str, audience: str, sections: List[str], prefs: UserPreferences
    ) -> Deck:
        system = (
            "You are a slide-writing agent. Honor the user's stored preferences exactly. "
            + _GEN_SCHEMA
        )
        user = json.dumps(
            {
                "topic": topic,
                "audience": audience,
                "sections": sections,
                "preferences": prefs.to_dict(),
                "constraints": {
                    "max_bullets_per_slide": prefs.max_bullets_per_slide,
                    "tone": prefs.tone,
                    "avoid_topics": prefs.avoid_topics,
                },
            }
        )
        data = self._chat(system, user)
        slides = []
        for i, s in enumerate(data.get("slides", []), start=1):
            slides.append(
                Slide(
                    id=i,
                    title=s.get("title", f"Slide {i}"),
                    bullets=list(s.get("bullets", []))[: prefs.max_bullets_per_slide],
                    layout=s.get("layout", "title-bullets"),
                )
            )
        if not slides:  # never return an empty deck
            return MockBrain().generate_deck(topic, audience, sections, prefs)
        return Deck(topic=topic, theme=prefs.theme, audience=audience, tone=prefs.tone, slides=slides)

    def localize(self, feedback: str, deck: Deck) -> Dict[str, Any]:
        system = (
            "You localize slide feedback to the smallest possible edit. Prefer editing a "
            "single slide over regenerating the deck. " + _LOCALIZE_SCHEMA
        )
        user = json.dumps(
            {
                "feedback": feedback,
                "deck": {
                    "theme": deck.theme,
                    "tone": deck.tone,
                    "slides": [{"id": s.id, "title": s.title, "bullets": s.bullets} for s in deck.slides],
                },
            }
        )
        try:
            plan = self._chat(system, user)
        except Exception:
            return MockBrain().localize(feedback, deck)
        # normalize + fall back on malformed plans
        if not isinstance(plan, dict) or "op" not in plan:
            return MockBrain().localize(feedback, deck)
        plan.setdefault("scope", "slide")
        plan.setdefault("params", {})
        plan.setdefault("durable", False)
        plan.setdefault("preference", None)
        return plan


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def make_brain(provider: Optional[str], model: Optional[str], temperature: float = 0.2) -> Brain:
    """Factory. ``provider`` defaults to env or ``mock``."""
    provider = (provider or os.getenv("MEMSLIDES_PROVIDER") or "mock").lower()
    if provider == "mock":
        return MockBrain()
    default_models = {"openai": "gpt-4o-mini", "groq": "llama-3.1-8b-instant", "ollama": "llama3.1"}
    model = model or os.getenv("MEMSLIDES_MODEL") or default_models.get(provider, provider)
    return LLMBrain(provider=provider, model=model, temperature=temperature)
