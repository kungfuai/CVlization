from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from dotenv import load_dotenv
from llama_cpp import Llama

from scripts.download_models import ensure_models

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "source"
MODEL_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"
OUTPUT_DIR = BASE_DIR / "outputs"


def _load_env() -> None:
    candidates: List[Path] = []
    current = BASE_DIR
    for parent in current.parents:
        candidates.append(parent / ".env")
    candidates.append(Path("/cvlization_repo/.env"))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env()

THREADS = int(os.getenv("LLAMACPP_THREADS", str(os.cpu_count() or 4)))
CHUNK_CHARS = int(os.getenv("LLAMACPP_CHUNK_CHARS", "500"))
CHUNK_OVERLAP = int(os.getenv("LLAMACPP_CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("LLAMACPP_TOP_K", "4"))
TEMPERATURE = float(os.getenv("LLAMACPP_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("LLAMACPP_MAX_TOKENS", "512"))
CONTEXT_CHARS = int(os.getenv("LLAMACPP_CONTEXT_CHARS", "800"))


@dataclass
class RetrievalChunk:
    doc_id: str
    text: str
    score: float
    chunk_index: int


def ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_documents() -> Iterable[Tuple[str, str]]:
    for path in sorted(DATA_DIR.glob("*.txt")):
        yield (path.stem, path.read_text())


def _split_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    collapsed = "\n".join(line.strip() for line in text.strip().splitlines())
    if not collapsed:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(collapsed):
        end = min(len(collapsed), start + chunk_chars)
        chunk = collapsed[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(collapsed):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _load_embedding_model(model_path: Path) -> Llama:
    return Llama(
        model_path=str(model_path),
        embedding=True,
        n_threads=THREADS,
        n_ctx=2048,
    )


def _embed_texts(llm: Llama, texts: List[str]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for text in texts:
        raw = llm.embed(text)
        vectors.append(np.asarray(raw, dtype=np.float32))
    return np.stack(vectors, axis=0)


def build_index() -> Dict[str, Path]:
    ensure_storage_dir()
    model_paths = ensure_models()

    embed_model = _load_embedding_model(model_paths["embed"])

    chunks: List[dict] = []
    embeddings: List[np.ndarray] = []

    for doc_id, text in iter_documents():
        chunk_texts = _split_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
        if not chunk_texts:
            continue
        vectors = _embed_texts(embed_model, chunk_texts)
        embeddings.append(vectors)
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append({
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk_text,
            })

    if not chunks:
        raise RuntimeError("No documents were ingested; add .txt files under data/source/.")

    matrix = embeddings[0] if len(embeddings) == 1 else np.concatenate(embeddings, axis=0)
    np.savez(STORAGE_DIR / "index.npz", embeddings=matrix)
    (STORAGE_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2))
    return model_paths


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _load_index() -> Tuple[np.ndarray, List[dict]]:
    index_path = STORAGE_DIR / "index.npz"
    chunks_path = STORAGE_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Vector store missing. Run ingest.sh first.")
    embeddings = np.load(index_path)["embeddings"].astype(np.float32)
    chunk_meta = json.loads(chunks_path.read_text())
    return embeddings, chunk_meta


def _format_context(chunks: List[RetrievalChunk]) -> str:
    formatted = []
    for chunk in chunks:
        formatted.append(
            f"Source: {chunk.doc_id} (chunk {chunk.chunk_index}, score={chunk.score:.3f})\n{chunk.text}"
        )
    return "\n\n".join(formatted)


def _load_chat_model(model_path: Path) -> Llama:
    return Llama(
        model_path=str(model_path),
        n_threads=THREADS,
        n_ctx=max(2048, CONTEXT_CHARS * 2),
        logits_all=False,
    )


def run_query(question: str, top_k: int | None = None) -> Dict[str, object]:
    ensure_storage_dir()
    model_paths = ensure_models()

    embeddings, metadata = _load_index()
    embed_model = _load_embedding_model(model_paths["embed"])

    query_vector = np.asarray(embed_model.embed(question), dtype=np.float32)
    query_norm = query_vector / max(np.linalg.norm(query_vector), 1e-12)

    matrix = _normalize(embeddings)
    scores = matrix @ query_norm

    k = top_k or RETRIEVAL_TOP_K
    top_indices = np.argsort(scores)[-k:][::-1]
    selected_chunks: List[RetrievalChunk] = []
    for idx in top_indices:
        meta = metadata[idx]
        selected_chunks.append(
            RetrievalChunk(
                doc_id=meta["doc_id"],
                chunk_index=int(meta["chunk_index"]),
                text=meta["text"],
                score=float(scores[idx]),
            )
        )

    context = _format_context(selected_chunks)

    llm = _load_chat_model(model_paths["llm"])
    prompt = (
        "You are a concise financial analyst working entirely offline. "
        "Answer using only the supplied context. Cite Lyft or Uber figures when relevant. "
        "If the answer cannot be derived, respond with 'Insufficient context.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    response = llm.create_completion(
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    answer = response["choices"][0]["text"].strip()

    return {
        "question": question,
        "answer": answer,
        "retrieved": [chunk.__dict__ for chunk in selected_chunks],
    }
