from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

HELPDESK_COLLECTION = "helpdesk_docs"
DEFAULT_EMBED_MODEL = os.getenv(
    "HELPDESK_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
PROVIDER_ENV_KEYS = (
    "HELPDESK_LLM_PROVIDER",
    "HELPDESK_LLM",
    "LLM_PROVIDER",
)
MODEL_ENV_KEYS = (
    "HELPDESK_LLM_MODEL",
    "HELPDESK_MODEL",
)
TEMP_ENV_KEYS = (
    "HELPDESK_LLM_TEMPERATURE",
    "HELPDESK_TEMPERATURE",
)


class RAGState(TypedDict, total=False):
    question: str
    context_docs: List[Document]
    answer: str
    citations: List[Dict[str, Any]]


def get_embedding_model(cache_dir: Path | None = None) -> HuggingFaceEmbeddings:
    kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_folder"] = str(cache_dir)
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL, **kwargs)


def build_retriever(
    persist_directory: Path, top_k: int = 4
) -> Runnable[[str], List[Document]]:
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. Run ingest.py first."
        )
    embedding = get_embedding_model(cache_dir=Path("var") / "embeddings")
    vectordb = Chroma(
        collection_name=HELPDESK_COLLECTION,
        embedding_function=embedding,
        persist_directory=str(persist_directory),
    )
    return vectordb.as_retriever(search_kwargs={"k": top_k})


def load_source_documents(
    docs_path: Path, glob: str = "**/*"
) -> List[Document]:
    if not docs_path.exists():
        raise FileNotFoundError(f"Expected docs directory at {docs_path}")
    loader = DirectoryLoader(
        str(docs_path),
        glob=glob,
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)


def _get_env_value(keys: tuple[str, ...], default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def get_llm(
    provider_override: str | None = None,
    model_override: str | None = None,
    temperature_override: float | None = None,
) -> tuple[BaseChatModel | None, str]:
    provider = (
        provider_override
        or _get_env_value(PROVIDER_ENV_KEYS, default="mock")
        or "mock"
    ).lower()
    model = model_override or _get_env_value(MODEL_ENV_KEYS)
    temperature = (
        temperature_override
        if temperature_override is not None
        else float(_get_env_value(TEMP_ENV_KEYS, default="0.0") or "0.0")
    )

    if provider in {"mock", "fake"}:
        return None, provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = model or "mistral"
        return ChatOllama(model=model_name, temperature=temperature), provider
    if provider in {"openai", "gpt"}:
        from langchain_openai import ChatOpenAI

        model_name = model or "gpt-4o-mini"
        return ChatOpenAI(model=model_name, temperature=temperature), provider
    if provider == "groq":
        from langchain_groq import ChatGroq

        model_name = model or "llama3-8b-8192"
        return ChatGroq(model=model_name, temperature=temperature), provider

    raise ValueError(
        f"Unsupported HELPDSK_LLM_PROVIDER/HELPDESK_LLM/LLM_PROVIDER '{provider}'. "
        "Supported providers: mock, ollama, openai, groq."
    )


def build_graph(
    retriever: Runnable[[str], List[Document]],
    llm: BaseChatModel | None,
    provider: str,
):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful technical support agent for CVlization. "
                    "Use the provided context to answer the user question. "
                    "If the answer cannot be found, say you do not know."
                ),
            ),
            (
                "human",
                (
                    "Context:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Provide a concise answer and cite relevant documents "
                    "using [doc:index] notation."
                ),
            ),
        ]
    )
    parser = StrOutputParser() if llm else None
    chain = qa_prompt | llm | parser if llm else None

    def retrieve_node(state: RAGState, **kwargs: Any) -> RAGState:
        question = state["question"]
        docs = retriever.invoke(question)
        return {"context_docs": docs}

    def answer_node(state: RAGState, **kwargs: Any) -> RAGState:
        docs = state.get("context_docs", [])
        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            context_blocks.append(f"[doc:{idx}] {doc.page_content}")
            citations.append(
                {
                    "doc": idx,
                    "source": doc.metadata.get("source"),
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                }
            )
        context_str = "\n\n".join(context_blocks) if context_blocks else "No context"
        if chain is None:
            if docs:
                top = docs[0]
                snippet = ""
                for line in top.page_content.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    snippet = stripped
                    break
                if not snippet:
                    snippet = top.page_content.strip().splitlines()[0]
                answer = (
                    f"{snippet[:350]} "
                    + (f"[doc:1]" if citations else "")
                )
            else:
                answer = "I do not have enough information to answer that question."
        else:
            answer = chain.invoke({"context": context_str, "question": state["question"]})
        return {"answer": answer, "citations": citations}

    graph_builder = StateGraph(RAGState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", answer_node)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile()
