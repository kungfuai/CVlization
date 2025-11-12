#!/usr/bin/env python3
"""
Create subset of CheckboxQA for testing/development.

Usage:
    # Create small dev set (first 5 docs)
    python3 create_subset.py --num-docs 5 --output data/subset_dev.jsonl

    # Create random subset
    python3 create_subset.py --num-docs 10 --random --output data/subset_10.jsonl

    # Create subset with specific document IDs
    python3 create_subset.py --doc-ids e5076219 0ad24b57 --output data/subset_custom.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_gold(gold_path: Path) -> List[dict]:
    """Load gold.jsonl into list of documents."""
    documents = []
    with open(gold_path, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def save_subset(documents: List[dict], output_path: Path):
    """Save subset to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    total_questions = sum(len(doc['annotations']) for doc in documents)
    print(f"✓ Saved {len(documents)} documents with {total_questions} questions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create CheckboxQA subset')
    parser.add_argument('--gold', type=Path, default=Path('data/gold.jsonl'),
                        help='Path to full gold.jsonl')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output path for subset')
    parser.add_argument('--num-docs', type=int,
                        help='Number of documents in subset')
    parser.add_argument('--doc-ids', nargs='+',
                        help='Specific document IDs to include')
    parser.add_argument('--random', action='store_true',
                        help='Select documents randomly (default: first N)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if not args.gold.exists():
        print(f"Error: Gold file not found: {args.gold}")
        return 1

    # Load all documents
    print(f"Loading from {args.gold}...")
    all_docs = load_gold(args.gold)
    total_q = sum(len(doc['annotations']) for doc in all_docs)
    print(f"Loaded {len(all_docs)} documents with {total_q} questions")

    # Select subset
    if args.doc_ids:
        # Select specific document IDs
        doc_id_set = set(args.doc_ids)
        subset = [doc for doc in all_docs if doc['name'] in doc_id_set]

        if len(subset) != len(args.doc_ids):
            found = {doc['name'] for doc in subset}
            missing = doc_id_set - found
            print(f"Warning: Could not find documents: {missing}")

        print(f"Selected {len(subset)} documents by ID")

    elif args.num_docs:
        if args.random:
            # Random sample
            random.seed(args.seed)
            subset = random.sample(all_docs, min(args.num_docs, len(all_docs)))
            print(f"Selected {len(subset)} random documents (seed={args.seed})")
        else:
            # First N documents
            subset = all_docs[:args.num_docs]
            print(f"Selected first {len(subset)} documents")

    else:
        print("Error: Must specify either --num-docs or --doc-ids")
        return 1

    # Save subset
    save_subset(subset, args.output)

    # Print sample
    if subset:
        print(f"\nSample document IDs:")
        for doc in subset[:5]:
            num_q = len(doc['annotations'])
            print(f"  - {doc['name']} ({num_q} questions)")
        if len(subset) > 5:
            print(f"  ... and {len(subset) - 5} more")

    return 0


if __name__ == "__main__":
    exit(main())
