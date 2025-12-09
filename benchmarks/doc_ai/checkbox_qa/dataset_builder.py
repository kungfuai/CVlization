#!/usr/bin/env python3
"""
CheckboxQA Dataset Builder

Loads the CheckboxQA dataset from HuggingFace or local JSONL files.
Provides a simple interface for accessing questions and ground truth answers.

DEPRECATED: This module is maintained for backward compatibility.
Use the new checkbox_qa package instead:

    from checkbox_qa import load_checkbox_qa
    dataset = load_checkbox_qa()

The new package provides:
- Automatic download and caching to ~/.cache/cvlization/data/checkbox_qa/
- HuggingFace datasets-like API
- PDF download support
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Try to import from the new checkbox_qa package
try:
    from checkbox_qa import get_cache_dir as _get_cache_dir
except ImportError:
    _get_cache_dir = None


def get_default_cache_dir() -> Path:
    """Get the default cache directory for CheckboxQA."""
    if _get_cache_dir:
        return _get_cache_dir()
    # Fallback if checkbox_qa package not available
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization" / "data" / "checkbox_qa"


@dataclass
class Question:
    """A single question from CheckboxQA."""

    id: int
    question: str
    answers: List[str]  # Ground truth answers (including variants)
    document_id: str


@dataclass
class Document:
    """A document with associated questions."""

    document_id: str
    pdf_path: Optional[Path]
    questions: List[Question]


class CheckboxQADataset:
    """
    CheckboxQA Dataset loader.

    Supports loading from:
    1. HuggingFace datasets (mturski/CheckboxQA)
    2. Local JSONL file (data/gold.jsonl or cached files)

    NOTE: Consider using the new checkbox_qa package for automatic download:

        from checkbox_qa import load_checkbox_qa
        dataset = load_checkbox_qa()
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = None,
        use_hf: bool = False,
        split: str = "test"
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing local data files.
                     Default: ~/.cache/cvlization/data/checkbox_qa/ or ./data/
            use_hf: If True, load from HuggingFace; else load from local files (default: False)
                   Note: HuggingFace version requires newer datasets library with Pdf feature support
            split: Dataset split ('test' only for CheckboxQA)
        """
        # Use cache directory by default, fall back to ./data/ for backward compatibility
        if data_dir is None:
            cache_dir = get_default_cache_dir()
            if (cache_dir / "gold.jsonl").exists():
                data_dir = cache_dir
            else:
                data_dir = Path("data")

        self.data_dir = Path(data_dir)
        self.use_hf = use_hf
        self.split = split
        self.documents: List[Document] = []

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from HuggingFace or local files."""
        if self.use_hf:
            self._load_from_huggingface()
        else:
            self._load_from_local()

    def _load_from_huggingface(self):
        """Load dataset from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )

        print("Loading CheckboxQA from HuggingFace...")
        dataset = load_dataset("mturski/CheckboxQA", split=self.split)

        for item in dataset:
            questions = []
            for annotation in item["annotations"]:
                # Collect all answer variants
                answers = []
                for value_dict in annotation["values"]:
                    answers.append(value_dict["value"])
                    if "value_variants" in value_dict:
                        answers.extend(value_dict["value_variants"])

                questions.append(Question(
                    id=annotation["id"],
                    question=annotation["key"],
                    answers=answers,
                    document_id=item["name"]
                ))

            # Check if PDF exists locally
            pdf_path = self.data_dir / "documents" / f"{item['name']}.{item['extension']}"
            if not pdf_path.exists():
                pdf_path = None

            self.documents.append(Document(
                document_id=item["name"],
                pdf_path=pdf_path,
                questions=questions
            ))

        print(f"Loaded {len(self.documents)} documents with {self.total_questions()} questions")

    def _load_from_local(self):
        """Load dataset from local JSONL file."""
        gold_path = self.data_dir / "gold.jsonl"

        if not gold_path.exists():
            # Try cache directory as fallback
            cache_dir = get_default_cache_dir()
            cache_gold = cache_dir / "gold.jsonl"
            if cache_gold.exists():
                self.data_dir = cache_dir
                gold_path = cache_gold
            else:
                raise FileNotFoundError(
                    f"Gold file not found at {gold_path} or {cache_gold}.\n"
                    "Download the dataset with:\n"
                    "  python -m checkbox_qa.dataset --download-only\n"
                    "Or use: from checkbox_qa import load_checkbox_qa; load_checkbox_qa()"
                )

        print(f"Loading CheckboxQA from {gold_path}...")
        self.documents = self._load_documents_from_jsonl(gold_path, self.data_dir / "documents")
        print(f"Loaded {len(self.documents)} documents with {self.total_questions()} questions")

    @classmethod
    def from_jsonl(cls, jsonl_path: Union[str, Path], data_dir: Union[str, Path] = None) -> "CheckboxQADataset":
        """Create a dataset instance from a specific JSONL subset."""
        dataset = cls.__new__(cls)
        if data_dir is None:
            data_dir = get_default_cache_dir()
        dataset.data_dir = Path(data_dir)
        dataset.use_hf = False
        dataset.split = "subset"
        dataset.documents = dataset._load_documents_from_jsonl(Path(jsonl_path), dataset.data_dir / "documents")
        return dataset

    def _load_documents_from_jsonl(self, jsonl_path: Path, documents_dir: Path) -> List[Document]:
        docs: List[Document] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                questions = []
                for annotation in item["annotations"]:
                    answers = []
                    for value_dict in annotation["values"]:
                        answers.append(value_dict["value"])
                        if "value_variants" in value_dict:
                            answers.extend(value_dict["value_variants"])
                    questions.append(Question(
                        id=annotation["id"],
                        question=annotation["key"],
                        answers=answers,
                        document_id=item["name"]
                    ))

                pdf_path = documents_dir / f"{item['name']}.{item.get('extension', 'pdf')}"
                if not pdf_path.exists():
                    pdf_path = None

                docs.append(Document(
                    document_id=item["name"],
                    pdf_path=pdf_path,
                    questions=questions
                ))
        return docs

    def total_questions(self) -> int:
        """Get total number of questions across all documents."""
        return sum(len(doc.questions) for doc in self.documents)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None

    def __len__(self) -> int:
        """Number of documents in the dataset."""
        return len(self.documents)

    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("CheckboxQA Dataset Builder")
    print("=" * 60)

    # Try loading from local files (default)
    try:
        dataset = CheckboxQADataset(use_hf=False)
        print(f"\nDataset loaded successfully!")
        print(f"Total documents: {len(dataset)}")
        print(f"Total questions: {dataset.total_questions()}")

        # Show first document
        if dataset.documents:
            doc = dataset.documents[0]
            print(f"\nFirst document: {doc.document_id}")
            print(f"PDF path: {doc.pdf_path}")
            print(f"Questions: {len(doc.questions)}")
            if doc.questions:
                q = doc.questions[0]
                print(f"\nFirst question:")
                print(f"  Q: {q.question}")
                print(f"  A: {q.answers[0]}")
                if len(q.answers) > 1:
                    print(f"  Variants: {q.answers[1:]}")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Make sure you have: pip install datasets")
