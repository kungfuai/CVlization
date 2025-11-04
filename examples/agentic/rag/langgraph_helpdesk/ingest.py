from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline import HELPDESK_COLLECTION, build_retriever, get_embedding_model, load_source_documents
from langchain_community.vectorstores import Chroma


def ingest(docs_dir: Path, persist_dir: Path, reset: bool = False) -> None:
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = load_source_documents(docs_dir)
    embedding = get_embedding_model(cache_dir=Path("var") / "embeddings")

    Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=HELPDESK_COLLECTION,
        persist_directory=str(persist_dir),
    )

    # Touch retriever to confirm persistence works
    build_retriever(persist_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest helpdesk documents into Chroma vector store."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("data") / "docs",
        help="Directory containing markdown or text files to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("var") / "chroma",
        help="Directory to store the Chroma collection.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing vector store before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(args.docs_dir, args.persist_dir, args.reset)
    print(f"Ingested documents from {args.docs_dir} into {args.persist_dir}.")


if __name__ == "__main__":
    main()
