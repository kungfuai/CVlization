"""Build-time grounding patches for the installed SearchOS package.

Two surgical patches, both motivated by reproducible hallucination against
the bundled offline corpus (upstream commit 361373b, sub-agent model
gpt-4.1; upstream's default gpt-5.5 is presumably more robust):

1. **Search-agent persona** (``searchos/agents/search/agent.md``): the
   opening Role paragraph says "End by emitting a natural-language assistant
   message with no tool call". gpt-4.1-class models read that as the
   *immediate* instruction and answer in a single turn from prior knowledge
   — zero tool calls, fabricated values, fabricated citation trail
   (reproduced 6/6 with the stock prompt). Both "no tool call" clauses are
   reworded to make tool-first sequencing explicit; with the reword the same
   model uses tools 6/6 and every value carries a real page citation.

2. **Final-summary evidence replay** (``evidence_extraction.py``): every
   sub-agent's final message is unconditionally replayed into the evidence
   graph as an ``agent://<id>/final_summary`` observation. When an agent
   hallucinates its summary, the fabricated values become coverage-map
   evidence with no page backing. The patch adds an env-var gate,
   ``SF_ENABLE_SUMMARY_REPLAY`` (default ``true`` = stock behavior);
   ``run_search.py`` sets it to ``false`` so cells can only be filled from
   anchored page-open evidence and un-grounded cells stay open for
   re-dispatch instead of silently absorbing a guess.

Exact-match replacement: if upstream changes either file, this script exits
non-zero and the Docker build fails, so the patches cannot rot silently.
"""

import sys
from pathlib import Path

import searchos.agents.search as search_role
import searchos.harness.middleware.extraction.evidence_extraction as evx

PERSONA_REPLACEMENTS = [
    (
        "End by emitting a natural-language assistant message with no tool "
        "call — the Orchestrator reads that message verbatim to decide what "
        "to dispatch next.",
        "Always begin by gathering evidence with your tools — your first "
        "action in every dispatch is a `search` call; never answer from "
        "prior knowledge. Only after you have opened the relevant pages, end "
        "by emitting a natural-language assistant message with no tool call "
        "— the Orchestrator reads that message verbatim to decide what to "
        "dispatch next.",
    ),
    (
        "End with a natural-language summary, no tool call — even on failure.",
        "After your tool-driven investigation (never before your first tool "
        "call), end with a natural-language summary, no tool call — even on "
        "failure.",
    ),
]


EVIDENCE_REPLACEMENTS = [
    (
        "    def _final_message_observation(self, response: Any) -> "
        "EvidenceObservation | None:\n"
        "        message = unwrap_ai_message(response)",
        "    def _final_message_observation(self, response: Any) -> "
        "EvidenceObservation | None:\n"
        "        import os as _os\n"
        '        if _os.environ.get("SF_ENABLE_SUMMARY_REPLAY", "true")'
        '.lower() in ("0", "false", "no"):\n'
        "            return None\n"
        "        message = unwrap_ai_message(response)",
    ),
]


def _patch(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            print(
                f"patch_search_persona: expected text not found in {path}:\n"
                f"  {old[:80]}...\n"
                "Upstream changed — re-verify whether the patch is still "
                "needed and update the replacements.",
                file=sys.stderr,
            )
            return False
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(f"patch_search_persona: patched {path}")
    return True


def main() -> int:
    persona = Path(search_role.__file__).parent / "agent.md"
    if not _patch(persona, PERSONA_REPLACEMENTS):
        return 1
    if not _patch(Path(evx.__file__), EVIDENCE_REPLACEMENTS):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
