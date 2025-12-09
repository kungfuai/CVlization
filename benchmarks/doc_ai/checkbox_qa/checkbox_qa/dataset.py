#!/usr/bin/env python3
"""CheckboxQA Dataset Loader with automatic download and caching.

This module provides a HuggingFace datasets-like interface for the CheckboxQA
benchmark dataset. It automatically downloads and caches the dataset on first use.

Cache location:
    Default: ~/.cache/cvlization/data/checkbox_qa/
    Override: Set CHECKBOX_QA_CACHE_DIR environment variable
"""

import json
import os
from dataclasses import dataclass
from logging import getLogger, warning
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

logger = getLogger(__name__)

# Default cache directory follows CVlization conventions
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cvlization" / "data" / "checkbox_qa"

# HuggingFace dataset identifier
HF_DATASET_ID = "mturski/CheckboxQA"

# Official CheckboxQA GitHub repository (for document URL map)
CHECKBOX_QA_GITHUB_RAW = "https://raw.githubusercontent.com/Snowflake-Labs/CheckboxQA/main"


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
    extension: str = "pdf"


class CheckboxQADataset:
    """
    CheckboxQA Dataset with automatic download and caching.

    Supports loading from:
    1. HuggingFace datasets (mturski/CheckboxQA) - auto-downloads on first use
    2. Local JSONL files (cached or pre-downloaded)

    Examples:
        # Load full dataset
        dataset = CheckboxQADataset()

        # Load specific subset
        dataset = CheckboxQADataset(subset="dev")

        # Iterate over documents
        for doc in dataset:
            print(f"{doc.document_id}: {len(doc.questions)} questions")

        # Access specific document
        doc = dataset.get_document("a10f421b")
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        subset: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            cache_dir: Cache directory. Default: ~/.cache/cvlization/data/checkbox_qa/
                      Can also be set via CHECKBOX_QA_CACHE_DIR env var.
            subset: Which subset to load: "dev", "test", or None for full dataset.
                   - "dev": 5 docs, 40 questions (quick testing)
                   - "test": 20 docs, 138 questions (evaluation subset)
                   - None: Full 88 docs, 579 questions
            download: If True, download dataset if not cached. Default: True.
            force_download: If True, re-download even if cached. Default: False.
        """
        self.cache_dir = get_cache_dir(cache_dir)
        self.subset = subset
        self.documents: List[Document] = []

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Download if needed
        if download or force_download:
            download_dataset(self.cache_dir, force=force_download)

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from cached JSONL files."""
        if self.subset:
            jsonl_path = self.cache_dir / f"subset_{self.subset}.jsonl"
            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"Subset '{self.subset}' not found at {jsonl_path}. "
                    f"Available subsets: dev, test. Or use subset=None for full dataset."
                )
        else:
            jsonl_path = self.cache_dir / "gold.jsonl"
            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {jsonl_path}. "
                    "Run download_dataset() or set download=True."
                )

        logger.info(f"Loading CheckboxQA from {jsonl_path}...")
        self.documents = self._load_documents_from_jsonl(jsonl_path)
        logger.info(
            f"Loaded {len(self.documents)} documents with {self.total_questions()} questions"
        )

    def _load_documents_from_jsonl(self, jsonl_path: Path) -> List[Document]:
        """Load documents from a JSONL file."""
        docs: List[Document] = []
        documents_dir = self.cache_dir / "documents"

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                questions = []
                for annotation in item["annotations"]:
                    answers = []
                    for value_dict in annotation["values"]:
                        answers.append(value_dict["value"])
                        if "value_variants" in value_dict and value_dict["value_variants"]:
                            answers.extend(value_dict["value_variants"])
                    questions.append(
                        Question(
                            id=annotation["id"],
                            question=annotation["key"],
                            answers=answers,
                            document_id=item["name"],
                        )
                    )

                extension = item.get("extension", "pdf")
                pdf_path = documents_dir / f"{item['name']}.{extension}"
                if not pdf_path.exists():
                    pdf_path = None

                docs.append(
                    Document(
                        document_id=item["name"],
                        pdf_path=pdf_path,
                        questions=questions,
                        extension=extension,
                    )
                )
        return docs

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
    ) -> "CheckboxQADataset":
        """Create a dataset instance from a specific JSONL file.

        Args:
            jsonl_path: Path to the JSONL file
            data_dir: Directory containing documents/ folder (default: cache dir)

        Returns:
            CheckboxQADataset instance
        """
        dataset = cls.__new__(cls)
        dataset.cache_dir = get_cache_dir(data_dir)
        dataset.subset = None
        dataset.documents = dataset._load_documents_from_jsonl(Path(jsonl_path))
        return dataset

    def total_questions(self) -> int:
        """Get total number of questions across all documents."""
        return sum(len(doc.questions) for doc in self.documents)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None

    def documents_with_pdfs(self) -> List[Document]:
        """Get only documents that have downloaded PDFs."""
        return [doc for doc in self.documents if doc.pdf_path is not None]

    def missing_pdfs(self) -> List[str]:
        """Get document IDs that don't have downloaded PDFs."""
        return [doc.document_id for doc in self.documents if doc.pdf_path is None]

    def __len__(self) -> int:
        """Number of documents in the dataset."""
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        """Iterate over documents."""
        return iter(self.documents)

    def __getitem__(self, idx: int) -> Document:
        """Get document by index."""
        return self.documents[idx]


def get_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the cache directory for CheckboxQA dataset.

    Priority:
    1. Explicit cache_dir argument
    2. CHECKBOX_QA_CACHE_DIR environment variable
    3. Default: ~/.cache/cvlization/data/checkbox_qa/

    Args:
        cache_dir: Optional explicit cache directory

    Returns:
        Path to cache directory
    """
    if cache_dir:
        return Path(cache_dir)

    env_cache = os.environ.get("CHECKBOX_QA_CACHE_DIR")
    if env_cache:
        return Path(env_cache)

    # Use XDG_CACHE_HOME if set, otherwise ~/.cache
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "cvlization" / "data" / "checkbox_qa"


def download_dataset(
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    download_pdfs: bool = True,
) -> Path:
    """
    Download the CheckboxQA dataset from HuggingFace.

    Downloads:
    - gold.jsonl: Full annotations (579 questions across 88 documents)
    - document_url_map.json: PDF download URLs
    - subset_dev.jsonl: Dev subset (5 docs, 40 questions)
    - subset_test.jsonl: Test subset (20 docs, 138 questions)
    - documents/*.pdf: PDF documents (if download_pdfs=True)

    Args:
        cache_dir: Cache directory. Default: ~/.cache/cvlization/data/checkbox_qa/
        force: Re-download even if files exist. Default: False.
        download_pdfs: Also download PDF documents. Default: True.

    Returns:
        Path to cache directory
    """
    cache_path = get_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    gold_path = cache_path / "gold.jsonl"
    url_map_path = cache_path / "document_url_map.json"

    # Check if already downloaded
    if gold_path.exists() and url_map_path.exists() and not force:
        logger.info(f"Dataset already cached at {cache_path}")
        if download_pdfs:
            _download_pdfs(cache_path, force=False)
        return cache_path

    print(f"Downloading CheckboxQA dataset to {cache_path}...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required for download. "
            "Install with: pip install datasets"
        )

    # Load from HuggingFace
    # Note: This dataset uses a custom Pdf feature type that stores PDFs as bytes
    # We access the raw Arrow data to avoid decoding issues
    print(f"Loading from HuggingFace: {HF_DATASET_ID}...")
    try:
        from datasets import load_dataset
        hf_dataset = load_dataset(HF_DATASET_ID, split="test")
        # Access raw Arrow table to avoid Pdf decoding issues
        arrow_table = hf_dataset.data
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset from HuggingFace: {e}\n"
            "The dataset may require a newer version of the datasets library "
            "or special handling for the Pdf feature type."
        )

    # Convert to JSONL format and extract PDFs
    print("Converting to JSONL format...")
    document_urls: Dict[str, dict] = {}

    # Create documents directory for PDFs
    documents_dir = cache_path / "documents"
    documents_dir.mkdir(exist_ok=True)

    # Process each row from Arrow table directly to create gold.jsonl
    num_rows = arrow_table.num_rows
    print(f"Processing {num_rows} documents...")

    try:
        from tqdm import tqdm
        row_iter = tqdm(range(num_rows), desc="Extracting annotations")
    except ImportError:
        row_iter = range(num_rows)

    with open(gold_path, "w") as f:
        for i in row_iter:
            row = arrow_table.slice(i, 1).to_pydict()
            # Extract values (they come as lists of length 1)
            item = {k: v[0] if v else None for k, v in row.items()}
            doc_name = item["name"]

            # Extract document info for gold.jsonl
            doc_record = {
                "name": doc_name,
                "extension": item.get("extension", "pdf"),
                "annotations": item["annotations"],
                "language": item.get("language", "en"),
                "split": item.get("split", "test"),
            }
            f.write(json.dumps(doc_record) + "\n")

    # Download document URL map from official GitHub repo
    print("Downloading document URL map from GitHub...")
    try:
        import requests
        url_map_url = f"{CHECKBOX_QA_GITHUB_RAW}/data/document_url_map.json"
        response = requests.get(url_map_url, timeout=30)
        response.raise_for_status()
        document_urls = response.json()
        with open(url_map_path, "w") as f:
            json.dump(document_urls, f, indent=2)
        print(f"Downloaded URL map with {len(document_urls)} documents")
    except Exception as e:
        print(f"Warning: Failed to download document URL map: {e}")
        # Create empty URL map as fallback
        with open(url_map_path, "w") as f:
            json.dump({}, f, indent=2)

    # Create subsets
    _create_subsets(cache_path)

    print(f"Dataset saved to {cache_path}")

    # Download PDFs
    if download_pdfs and document_urls:
        _download_pdfs(cache_path, force=force)

    return cache_path


def _create_subsets(cache_path: Path) -> None:
    """Create dev and test subsets from full dataset."""
    import random

    gold_path = cache_path / "gold.jsonl"
    if not gold_path.exists():
        return

    # Load all documents
    with open(gold_path, "r") as f:
        all_docs = [json.loads(line) for line in f]

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    shuffled = all_docs.copy()
    random.shuffle(shuffled)

    # Dev subset: first 5 documents
    dev_docs = shuffled[:5]
    dev_path = cache_path / "subset_dev.jsonl"
    with open(dev_path, "w") as f:
        for doc in dev_docs:
            f.write(json.dumps(doc) + "\n")

    dev_questions = sum(len(d["annotations"]) for d in dev_docs)
    print(f"Created subset_dev.jsonl: {len(dev_docs)} docs, {dev_questions} questions")

    # Test subset: 20 documents
    test_docs = shuffled[:20]
    test_path = cache_path / "subset_test.jsonl"
    with open(test_path, "w") as f:
        for doc in test_docs:
            f.write(json.dumps(doc) + "\n")

    test_questions = sum(len(d["annotations"]) for d in test_docs)
    print(f"Created subset_test.jsonl: {len(test_docs)} docs, {test_questions} questions")


def _download_pdfs(cache_path: Path, force: bool = False) -> None:
    """Download PDF documents from URLs in document_url_map.json."""
    url_map_path = cache_path / "document_url_map.json"
    if not url_map_path.exists():
        warning("No document_url_map.json found, skipping PDF download")
        return

    with open(url_map_path, "r") as f:
        url_map = json.load(f)

    if not url_map:
        warning("document_url_map.json is empty, skipping PDF download")
        return

    documents_dir = cache_path / "documents"
    documents_dir.mkdir(exist_ok=True)

    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "requests and tqdm required for PDF download. "
            "Install with: pip install requests tqdm"
        )

    # Optional: PyPDF2 for page extraction
    try:
        from PyPDF2 import PdfReader, PdfWriter

        has_pypdf2 = True
    except ImportError:
        has_pypdf2 = False
        warning("PyPDF2 not installed, will download full PDFs without page extraction")

    print(f"Downloading {len(url_map)} PDF documents...")

    for doc_id, info in tqdm(url_map.items(), desc="Downloading PDFs"):
        pdf_path = documents_dir / f"{doc_id}.pdf"

        # Skip if already exists
        if pdf_path.exists() and not force:
            continue

        pdf_url = info.get("pdf_url")
        if not pdf_url:
            warning(f"No URL for document {doc_id}")
            continue

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Save PDF
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            # Extract specific pages if specified
            pages = info.get("pages")
            if pages and has_pypdf2:
                _extract_pages(pdf_path, pages)

        except Exception as e:
            warning(f"Failed to download {doc_id}: {e}")


def _extract_pages(pdf_path: Path, pages: List[int]) -> None:
    """Extract specific pages from a PDF (1-indexed)."""
    from PyPDF2 import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for page_num in pages:
        # Page numbers are 1-indexed in the dataset
        writer.add_page(reader.pages[page_num - 1])

    with open(pdf_path, "wb") as f:
        writer.write(f)


def load_checkbox_qa(
    subset: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> CheckboxQADataset:
    """
    Load the CheckboxQA dataset.

    This is the main entry point for loading the dataset. It automatically
    downloads and caches the dataset on first use.

    Args:
        subset: Which subset to load:
               - "dev": 5 docs, 40 questions (quick testing)
               - "test": 20 docs, 138 questions (evaluation subset)
               - None: Full 88 docs, 579 questions
        cache_dir: Cache directory. Default: ~/.cache/cvlization/data/checkbox_qa/
        download: If True, download dataset if not cached. Default: True.

    Returns:
        CheckboxQADataset instance

    Examples:
        # Load full dataset
        dataset = load_checkbox_qa()

        # Load dev subset for quick testing
        dataset = load_checkbox_qa(subset="dev")

        # Load from custom cache location
        dataset = load_checkbox_qa(cache_dir="/path/to/cache")

        # Iterate over documents
        for doc in dataset:
            print(f"{doc.document_id}: {len(doc.questions)} questions")
            if doc.pdf_path:
                print(f"  PDF: {doc.pdf_path}")
    """
    return CheckboxQADataset(
        cache_dir=cache_dir,
        subset=subset,
        download=download,
    )


if __name__ == "__main__":
    # Example usage / CLI
    import argparse

    parser = argparse.ArgumentParser(description="CheckboxQA Dataset Loader")
    parser.add_argument(
        "--subset",
        choices=["dev", "test"],
        help="Load a subset instead of full dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: ~/.cache/cvlization/data/checkbox_qa/)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Just download the dataset, don't load it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--no-pdfs",
        action="store_true",
        help="Skip downloading PDF documents",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CheckboxQA Dataset Loader")
    print("=" * 60)

    cache_dir = get_cache_dir(args.cache_dir)
    print(f"Cache directory: {cache_dir}")

    if args.download_only:
        download_dataset(
            cache_dir=cache_dir,
            force=args.force,
            download_pdfs=not args.no_pdfs,
        )
    else:
        dataset = load_checkbox_qa(
            subset=args.subset,
            cache_dir=cache_dir,
        )
        print(f"\nDataset loaded:")
        print(f"  Documents: {len(dataset)}")
        print(f"  Questions: {dataset.total_questions()}")
        print(f"  With PDFs: {len(dataset.documents_with_pdfs())}")

        missing = dataset.missing_pdfs()
        if missing:
            print(f"  Missing PDFs: {len(missing)}")

        if dataset.documents:
            doc = dataset.documents[0]
            print(f"\nFirst document: {doc.document_id}")
            print(f"  PDF: {doc.pdf_path}")
            print(f"  Questions: {len(doc.questions)}")
            if doc.questions:
                q = doc.questions[0]
                print(f"\n  Sample question:")
                print(f"    Q: {q.question}")
                print(f"    A: {q.answers[0]}")
