from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

STORAGE_DIR = Path(__file__).resolve().parent / "storage"
HF_CACHE = Path(os.getenv("HF_HOME", "/workspace/.cache/huggingface"))

DEFAULT_PROVIDER = os.getenv("LLAMA_REACT_PROVIDER", "mock").lower()
DEFAULT_OPENAI_MODEL = os.getenv("LLAMA_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_HF_MODEL = os.getenv("LLAMA_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEFAULT_LOCAL_MODEL = os.getenv("LLAMA_LOCAL_MODEL", "google/flan-t5-small")
DEFAULT_EMBED_MODEL = os.getenv(
    "LLAMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MAX_CONTEXT_CHARS = int(os.getenv("LLAMA_CONTEXT_CHARS", "2000"))


def _ensure_hf_cache() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))


def _embedding() -> HuggingFaceEmbedding:
    _ensure_hf_cache()
    embed = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.embed_model = embed
    return embed


def load_query_engines() -> Dict[str, QueryEngineTool]:
    engines: Dict[str, QueryEngineTool] = {}
    for company, description in [
        (
            "lyft",
            "Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        (
            "uber",
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
    ]:
        _embedding()
        storage = StorageContext.from_defaults(
            persist_dir=str(STORAGE_DIR / company)
        )
        index = load_index_from_storage(storage)
        engine = index.as_query_engine(similarity_top_k=3)
        tool = QueryEngineTool.from_defaults(
            query_engine=engine,
            name=f"{company}_10k",
            description=description,
        )
        engines[company] = tool
    return engines


def load_indexes() -> Dict[str, any]:
    _embedding()
    indexes: Dict[str, any] = {}
    for company in ["lyft", "uber"]:
        storage = StorageContext.from_defaults(
            persist_dir=str(STORAGE_DIR / company)
        )
        indexes[company] = load_index_from_storage(storage)
    return indexes


def _configure_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider 'openai'.")
        llm = OpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0.0)
        Settings.llm = llm
        return llm
    if provider in {"hf", "huggingface"}:
        _ensure_hf_cache()
        llm = HuggingFaceLLM(
            model_name=DEFAULT_HF_MODEL,
            tokenizer_name=DEFAULT_HF_MODEL,
            generate_kwargs={"max_new_tokens": 256},
        )
        Settings.llm = llm
        return llm
    raise ValueError(f"Unsupported provider '{provider}'.")


@dataclass
class AgentRunner:
    agent: ReActAgent
    context: Context

    async def aquery(self, question: str) -> str:
        run = self.agent.run(question, ctx=self.context)
        response = await run
        return str(response)

    def query(self, question: str) -> str:
        return asyncio.run(self.aquery(question))


class LocalContextFinanceAgent:
    def __init__(self, indexes: Dict[str, any]) -> None:
        self.retrievers = {
            name: index.as_retriever(similarity_top_k=3) for name, index in indexes.items()
        }

    def query(self, question: str) -> str:
        lines = [
            "Mock financial analysis (context retrieved without LLM).",
            f"Question: {question}",
            "",
        ]
        for name, retriever in self.retrievers.items():
            nodes = retriever.retrieve(question)
            snippets = []
            for node in nodes[:2]:
                content = node.node.get_content().strip().replace("\n", " ")
                if content:
                    snippets.append(content[:400])
            if not snippets:
                continue
            lines.append(f"{name.title()} context:")
            for snippet in snippets:
                lines.append(f"- {snippet}")
            lines.append("")
        if len(lines) <= 3:
            lines.append("No relevant context was retrieved.")
        return "\n".join(lines).strip()


def build_agent(provider: str | None = None):
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "openai":
        llm = _configure_llm(provider)
        tools = load_query_engines()
        agent = ReActAgent(
            tools=list(tools.values()),
            llm=llm,
        )
        ctx = Context(agent)
        return AgentRunner(agent=agent, context=ctx)

    if provider in {"hf", "huggingface"}:
        llm = _configure_llm(provider)
        tools = load_query_engines()
        agent = ReActAgent(
            tools=list(tools.values()),
            llm=llm,
        )
        ctx = Context(agent)
        return AgentRunner(agent=agent, context=ctx)

    indexes = load_indexes()
    return LocalContextFinanceAgent(indexes)
