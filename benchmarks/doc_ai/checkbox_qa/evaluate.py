#!/usr/bin/env python3
"""
CheckboxQA Evaluation Script

Evaluates model predictions against ground truth using ANLS* metric.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    return answer.lower().strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_levenshtein_similarity(pred: str, gold: str) -> float:
    """
    Compute Normalized Levenshtein Similarity (NLS).

    Returns a score between 0 and 1, where 1 is perfect match.
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0

    distance = levenshtein_distance(pred_norm, gold_norm)
    max_len = max(len(pred_norm), len(gold_norm))

    return 1.0 - (distance / max_len)


def anls_score_single(pred: str, gold_answers: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS score for a single prediction against multiple gold answers.

    Args:
        pred: Predicted answer
        gold_answers: List of acceptable gold answers (including variants)
        threshold: Minimum similarity threshold (default 0.5)

    Returns:
        Maximum ANLS score across all gold variants
    """
    if not gold_answers:
        return 0.0

    max_score = 0.0
    for gold in gold_answers:
        similarity = normalized_levenshtein_similarity(pred, gold)
        # Apply threshold: scores below threshold become 0
        if similarity >= threshold:
            max_score = max(max_score, similarity)

    return max_score


def read_jsonl(file_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Read JSONL file and extract QA pairs.

    Returns:
        Dict mapping document_id -> {question -> [answers]}
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
                    # Handle None/null values
                    if value_dict['value'] is None or value_dict['value'].lower() == 'none':
                        answers.append("")
                    else:
                        answers.append(value_dict['value'])

                    # Add variants
                    if 'value_variants' in value_dict:
                        answers.extend(value_dict['value_variants'])

                qas[question] = answers

            documents[doc['name']] = qas

    return documents


def evaluate_predictions(
    pred_dict: Dict[str, Dict[str, List[str]]],
    gold_dict: Dict[str, Dict[str, List[str]]],
    threshold: float = 0.5
) -> Tuple[float, int, int]:
    """
    Evaluate predictions against gold standard.

    Returns:
        (anls_score, total_questions, num_correct)
    """
    scores = []
    total_questions = 0
    num_correct = 0

    for doc_id, gold_qas in gold_dict.items():
        if doc_id not in pred_dict:
            print(f"Warning: No predictions for document {doc_id}", file=sys.stderr)
            # Add 0 scores for missing document
            scores.extend([0.0] * len(gold_qas))
            total_questions += len(gold_qas)
            continue

        pred_qas = pred_dict[doc_id]

        for question, gold_answers in gold_qas.items():
            total_questions += 1

            if question not in pred_qas:
                print(f"Warning: No prediction for question: {question[:50]}...", file=sys.stderr)
                scores.append(0.0)
                continue

            # Get first predicted answer (models typically return one answer)
            pred_answer = pred_qas[question][0] if pred_qas[question] else ""

            # Compute ANLS score
            score = anls_score_single(pred_answer, gold_answers, threshold)
            scores.append(score)

            if score >= threshold:
                num_correct += 1

    anls = sum(scores) / len(scores) if scores else 0.0
    return anls, total_questions, num_correct


def main():
    parser = argparse.ArgumentParser(description='Evaluate CheckboxQA predictions using ANLS*')
    parser.add_argument('--pred', type=Path, required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--gold', type=Path, default=Path('data/gold.jsonl'),
                        help='Path to gold standard JSONL file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='ANLS threshold (default: 0.5)')
    parser.add_argument('--output', type=Path,
                        help='Path to save detailed results')

    args = parser.parse_args()

    if not args.pred.exists():
        print(f"Error: Prediction file not found: {args.pred}", file=sys.stderr)
        sys.exit(1)

    if not args.gold.exists():
        print(f"Error: Gold file not found: {args.gold}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading predictions from {args.pred}")
    pred_dict = read_jsonl(args.pred)
    print(f"Loading gold standard from {args.gold}")
    gold_dict = read_jsonl(args.gold)

    # Evaluate
    anls, total_questions, num_correct = evaluate_predictions(
        pred_dict, gold_dict, args.threshold
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"ANLS* Score: {anls:.4f}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct (>= {args.threshold}): {num_correct} ({num_correct/total_questions*100:.1f}%)")
    print("=" * 60)

    # Save detailed results
    if args.output:
        results = {
            "anls_score": anls,
            "total_questions": total_questions,
            "num_correct": num_correct,
            "threshold": args.threshold,
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
