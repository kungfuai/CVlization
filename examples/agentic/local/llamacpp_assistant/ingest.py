from __future__ import annotations

from pipeline import build_index


def main() -> None:
    build_index()
    print("Ingestion complete. Embeddings stored under storage/.")


if __name__ == "__main__":
    main()
