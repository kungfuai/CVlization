from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI

HF_CACHE = Path(os.getenv("HF_HOME", "/workspace/.cache/huggingface"))
DEFAULT_PROVIDER = os.getenv("LLAMA_GRAPHRAG_PROVIDER", "mock").lower()
DEFAULT_OPENAI_MODEL = os.getenv("LLAMA_GRAPHRAG_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_HF_MODEL = os.getenv(
    "LLAMA_GRAPHRAG_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

PEOPLE: List[Dict[str, str]] = [
    {
        "name": "Jessica Miller",
        "title": "Experienced Sales Manager",
        "skills": "driving sales growth and building high-performing teams",
    },
    {
        "name": "David Thompson",
        "title": "Creative Graphic Designer",
        "skills": "visual design and branding",
    },
]


def _ensure_cache_dirs() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))


def _build_graph() -> nx.Graph:
    graph = nx.Graph()
    for person in PEOPLE:
        person_node = person["name"].lower()
        graph.add_node(person_node, type="person", title=person["title"], skills=person["skills"])
        graph.add_edge(person_node, "profession", relation=person["title"].lower())
        for keyword in person["skills"].split(" "):
            keyword = keyword.strip(".,").lower()
            if len(keyword) > 3:
                graph.add_edge(person_node, keyword, relation="skill")
    return graph


def _configure_llm(provider: str):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the GraphRAG example.")
        return OpenAI(model=DEFAULT_OPENAI_MODEL)
    if provider in {"hf", "huggingface"}:
        _ensure_cache_dirs()
        return HuggingFaceLLM(
            model_name=DEFAULT_HF_MODEL,
            tokenizer_name=DEFAULT_HF_MODEL,
            generate_kwargs={"max_new_tokens": 256},
        )
    raise ValueError(f"Unsupported provider '{provider}'.")


def _graph_context(graph: nx.Graph) -> str:
    chunks = []
    for person in PEOPLE:
        name = person["name"].lower()
        skills = person["skills"]
        title = person["title"]
        chunks.append(f"- {person['name']}: {title}; skills include {skills}.")
    return "\n".join(chunks)


def _generate_answer_with_llm(llm, question: str, graph: nx.Graph) -> str:
    context = _graph_context(graph)
    prompt = (
        "You are a graph reasoning assistant. The following bullet points come from a knowledge graph\n"
        "about professionals and their skills. Answer the question using only this information.\n\n"
        f"Graph facts:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return llm.predict(prompt).strip()


def _mock_answer(question: str) -> str:
    names = ", ".join(person["name"] for person in PEOPLE)
    return f"Mock answer: the people mentioned are {names}."


def run_query(question: str, provider: str | None = None) -> Dict[str, Any]:
    provider = (provider or DEFAULT_PROVIDER).lower()
    graph = _build_graph()

    if provider == "mock":
        answer = _mock_answer(question)
    else:
        llm = _configure_llm(provider)
        answer = _generate_answer_with_llm(llm, question, graph)

    rag_contexts = [json.dumps(person, indent=2) for person in PEOPLE]
    related_nodes = sorted(set(graph.neighbors("profession"))) if graph.has_node("profession") else []
    return {
        "mode": provider,
        "graph_answers": [answer],
        "rag_answers": rag_contexts,
        "related_nodes": related_nodes,
    }
