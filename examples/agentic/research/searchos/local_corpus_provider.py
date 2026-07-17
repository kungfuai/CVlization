"""A keyless, offline SearchProvider for the SearchOS example.

SearchOS binds its "web search" onto a small pluggable interface
(``searchos.tools.simple_browser.search.base.SearchProvider``): one async
``search(query, max_results)`` that returns ``SearchResult`` rows. Each row can
carry the full page ``content``; when the browser backend is set to
``search_engine`` (see ``run_search.py``), SearchOS serves ``open()`` calls
straight from that cached content and never touches the network.

That lets the *entire* multi-agent pipeline — Explore -> schema/coverage-map ->
pipelined sub-agents -> judge-based extraction -> synthesis — run end-to-end
against a fixed local corpus with **no search-API key**, which is what makes the
default demo mode reproducible in CI. Swap in Serper/Tavily (``--mode web``) and
the exact same pipeline runs against the live web instead.

Ranking is deliberately simple (lexical overlap with a light IDF weighting): the
point of the demo is the orchestration and SOCM state, not retrieval quality.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from searchos.tools.simple_browser.search.base import SearchProvider, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class LocalCorpusProvider(SearchProvider):
    """Serve search results from a bundled JSON corpus instead of the web."""

    def __init__(self, documents: list[dict]) -> None:
        if not documents:
            raise ValueError("LocalCorpusProvider needs a non-empty document list")
        self._docs = documents
        # Precompute per-doc token bags + a corpus-wide document frequency so
        # that generic words ("the", "model", "lab") count for less than the
        # discriminating entity names.
        self._doc_tokens: list[Counter] = []
        df: Counter = Counter()
        for doc in documents:
            toks = _tokenize(f"{doc.get('title', '')} {doc.get('content', '')}")
            bag = Counter(toks)
            self._doc_tokens.append(bag)
            for term in bag:
                df[term] += 1
        n = len(documents)
        self._idf = {term: math.log((n + 1) / (freq + 0.5)) for term, freq in df.items()}

    @classmethod
    def from_json(cls, path: str | Path) -> "LocalCorpusProvider":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = data["documents"] if isinstance(data, dict) else data
        return cls(docs)

    @property
    def name(self) -> str:
        return "local-corpus"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        q_terms = _tokenize(query)
        if not q_terms:
            ranked = list(range(len(self._docs)))
        else:
            scores: list[tuple[float, int]] = []
            for i, bag in enumerate(self._doc_tokens):
                score = sum(bag.get(t, 0) * self._idf.get(t, 0.0) for t in q_terms)
                if score > 0:
                    scores.append((score, i))
            scores.sort(key=lambda s: s[0], reverse=True)
            ranked = [i for _, i in scores]
            # Always surface something so a sub-agent never dead-ends on a query
            # that happens to miss every keyword.
            if not ranked:
                ranked = list(range(len(self._docs)))

        results: list[SearchResult] = []
        for i in ranked[: max(1, max_results)]:
            doc = self._docs[i]
            content = doc.get("content", "") or ""
            snippet = content[:200]
            results.append(
                SearchResult(
                    title=doc.get("title", "") or doc.get("url", ""),
                    url=doc.get("url", ""),
                    snippet=snippet,
                    content=content,
                    score=1.0,
                )
            )
        return results
