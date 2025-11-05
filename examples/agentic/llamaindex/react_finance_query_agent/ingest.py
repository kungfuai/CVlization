from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

from scripts.download_10k import FILES, download_file

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
STORAGE_DIR = BASE_DIR / "storage"


def _load_env() -> None:
    candidates = []
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / ".env")
    candidates.append(Path("/cvlization_repo/.env"))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env()

DEFAULT_EMBED_MODEL = os.getenv(
    "LLAMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def ensure_documents() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename in FILES:
        path = download_file(filename, FILES[filename])
        paths[filename] = path
    return paths


def build_index(name: str, file_path: Path) -> None:
    persist_dir = STORAGE_DIR / name
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"[ingest] Reusing existing index at {persist_dir}")
        return

    print(f"[ingest] Building index for {name} from {file_path}")
    embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    Settings.embed_model = embed_model

    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"[ingest] Saved index -> {persist_dir}")


def main() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    docs = ensure_documents()
    build_index("lyft", docs["lyft_2021.pdf"])
    build_index("uber", docs["uber_2021.pdf"])


if __name__ == "__main__":
    main()
