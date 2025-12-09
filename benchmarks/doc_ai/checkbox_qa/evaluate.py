#!/usr/bin/env python3
"""
CheckboxQA Evaluation Script

Evaluates model predictions against ground truth using ANLS* metric.
Uses the official anls_star library for compatibility with CheckboxQA paper.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import sys

try:
    from anls_star import anls_score
except ImportError:
    print("Error: anls_star library not found. Install with: pip install anls-star", file=sys.stderr)
    sys.exit(1)


def read_jsonl(file_path: Path) -> Dict[str, Dict[str, List[Union[str, Tuple[str], None]]]]:
    """
    Read JSONL file and extract QA pairs in CheckboxQA format.

    This matches the official CheckboxQA implementation:
    - value_variants are stored as tuples
    - None/null values are stored as None (not empty string)

    Returns:
        Dict mapping document_id -> {question -> [answers]}
        where answers can be strings, tuples (for variants), or None
    """
    documents = {}

    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            qas = {}

            for ann in doc['annotations']:
                question = ann['key']
                answers = []

                for value_dict in ann['values']:
                    # Handle None/null values - use None instead of empty string
                    if value_dict['value'] is None or value_dict['value'].lower() == 'none':
                        answers.append(None)
                    else:
                        # Store value_variants as tuple (official format)
                        if 'value_variants' in value_dict and value_dict['value_variants']:
                            answers.append(tuple([a for a in value_dict['value_variants']]))
                        else:
                            answers.append(value_dict['value'])

                qas[question] = answers

            documents[doc['name']] = qas

    return documents


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CheckboxQA predictions using ANLS* (official anls_star library)'
    )
    parser.add_argument('--pred', type=Path, required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--gold', type=Path, default=Path('data/gold.jsonl'),
                        help='Path to gold standard JSONL file')
    parser.add_argument('--output', type=Path,
                        help='Path to save detailed results (JSON)')

    args = parser.parse_args()

    if not args.pred.exists():
        print(f"Error: Prediction file not found: {args.pred}", file=sys.stderr)
        sys.exit(1)

    if not args.gold.exists():
        print(f"Error: Gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    # Load data in CheckboxQA format
    print(f"Loading predictions from {args.pred}")
    pred_dict = read_jsonl(args.pred)
    print(f"Loading gold standard from {args.gold}")
    gold_dict = read_jsonl(args.gold)

    # Compute ANLS* score using official library
    print("Computing ANLS* score using anls_star library...")
    anls = anls_score(gold_dict, pred_dict)

    # Count total questions
    total_questions = sum(len(qas) for qas in gold_dict.values())

    # Print results
    print("\n" + "=" * 60)
    print(f"ANLS* Score: {anls:.4f}")
    print(f"Total Questions: {total_questions}")
    print("=" * 60)

    # Save detailed results
    if args.output:
        results = {
            "anls_score": anls,
            "total_questions": total_questions,
            "prediction_file": str(args.pred),
            "gold_file": str(args.gold)
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

    return anls


if __name__ == '__main__':
    main()
