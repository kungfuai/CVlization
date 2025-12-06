"""CheckboxQA Dataset Loader.

A HuggingFace datasets-like module for loading the CheckboxQA benchmark dataset
with automatic download and caching.

Example usage:
    from checkbox_qa import load_checkbox_qa

    # Load full dataset (auto-downloads on first use)
    dataset = load_checkbox_qa()

    # Load a subset
    dataset = load_checkbox_qa(subset="dev")

    # Access documents
    for doc in dataset:
        print(f"Document: {doc.document_id}, Questions: {len(doc.questions)}")

Cache location:
    Default: ~/.cache/cvlization/data/checkbox_qa/
    Override: CHECKBOX_QA_CACHE_DIR environment variable
"""

from .dataset import (
    CheckboxQADataset,
    Document,
    Question,
    load_checkbox_qa,
    get_cache_dir,
    download_dataset,
)

__all__ = [
    "CheckboxQADataset",
    "Document",
    "Question",
    "load_checkbox_qa",
    "get_cache_dir",
    "download_dataset",
]

__version__ = "0.1.0"
