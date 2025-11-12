#!/usr/bin/env python3
"""
CheckboxQA Dataset Builder

Loads the CheckboxQA dataset from HuggingFace or local JSONL files.
Provides a simple interface for accessing questions and ground truth answers.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


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
    2. Local JSONL file (data/gold.jsonl)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        use_hf: bool = False,
        split: str = "test"
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing local data files
            use_hf: If True, load from HuggingFace; else load from local files (default: False)
                   Note: HuggingFace version requires newer datasets library with Pdf feature support
            split: Dataset split ('test' only for CheckboxQA)
        """
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
            raise FileNotFoundError(
                f"Gold file not found at {gold_path}. "
                "Either set use_hf=True or download the dataset locally."
            )

        print(f"Loading CheckboxQA from {gold_path}...")

        with open(gold_path, "r") as f:
            for line in f:
                item = json.loads(line)
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

                # Check if PDF exists
                pdf_path = self.data_dir / "documents" / f"{item['name']}.{item['extension']}"
                if not pdf_path.exists():
                    pdf_path = None

                self.documents.append(Document(
                    document_id=item["name"],
                    pdf_path=pdf_path,
                    questions=questions
                ))

        print(f"Loaded {len(self.documents)} documents with {self.total_questions()} questions")

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
