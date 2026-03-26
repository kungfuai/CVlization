#!/usr/bin/env python3
"""
Verification harness for AdderBoard-style submissions.
"""

import argparse
import importlib.util
import random
import time
from typing import Any, Dict, List, Tuple


MAX_INPUT = 9_999_999_999


def load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load submission from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model"):
        raise ValueError("Submission must define build_model() -> (model, metadata)")
    if not hasattr(mod, "add"):
        raise ValueError("Submission must define add(model, a, b) -> int")

    return mod


def _build_cases(num_tests: int, seed: int) -> List[Tuple[int, int]]:
    edge_cases = [
        (0, 0),
        (0, 1),
        (MAX_INPUT, 0),
        (MAX_INPUT, 1),
        (MAX_INPUT, MAX_INPUT),
        (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889),
        (1_234_567_890, 9_876_543_210),
        (MAX_INPUT, MAX_INPUT),
        (1, MAX_INPUT),
    ]

    rng = random.Random(seed)
    random_cases = [(rng.randint(0, MAX_INPUT), rng.randint(0, MAX_INPUT)) for _ in range(num_tests)]
    return edge_cases + random_cases


def run_test(mod, num_tests: int = 10000, seed: int = 2025, progress_every: int = 1000) -> Dict[str, Any]:
    model, metadata = mod.build_model()
    metadata = metadata or {}

    cases = _build_cases(num_tests=num_tests, seed=seed)
    total = len(cases)
    passed = 0
    failures: List[Tuple[int, int, int, Any]] = []

    start = time.time()
    for i, (a, b) in enumerate(cases):
        expected = a + b
        try:
            result = mod.add(model, a, b)
        except Exception as exc:  # pylint: disable=broad-except
            failures.append((a, b, expected, f"ERROR: {exc}"))
            continue

        if result == expected:
            passed += 1
        else:
            failures.append((a, b, expected, result))

        if progress_every > 0 and (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            print(f"  Progress: {i + 1}/{total} ({passed}/{i + 1} correct) [{elapsed:.1f}s]")

    elapsed = time.time() - start
    accuracy = (passed / total) * 100.0
    qualified = accuracy >= 99.0
    throughput = total / elapsed if elapsed > 0 else 0.0

    return {
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "qualified": qualified,
        "time": elapsed,
        "throughput": throughput,
        "metadata": metadata,
        "failures": failures,
    }


def _print_report(result: Dict[str, Any], max_failures: int = 20):
    metadata = result.get("metadata", {})
    print(f"Model: {metadata.get('name', 'unnamed')}")
    print(f"Author: {metadata.get('author', 'unknown')}")
    print(f"Parameters (unique): {metadata.get('params', '?')}")
    print(f"Architecture: {metadata.get('architecture', '?')}")
    print(f"Tricks: {', '.join(metadata.get('tricks', []))}")
    print()

    print(
        f"Results: {result['passed']}/{result['total']} correct "
        f"({result['accuracy']:.2f}%)"
    )
    print(
        f"Time: {result['time']:.1f}s "
        f"({result['throughput']:.0f} additions/sec)"
    )
    print(
        f"Status: {'QUALIFIED' if result['qualified'] else 'NOT QUALIFIED'} "
        "(threshold: 99%)"
    )

    failures = result.get("failures", [])
    if failures:
        shown = failures[:max_failures]
        print()
        if len(failures) > max_failures:
            print(f"First {max_failures} failures (of {len(failures)}):")
        else:
            print(f"Failures ({len(failures)}):")
        for a, b, expected, got in shown:
            print(f"  {a} + {b} = {expected}, got {got}")


def main():
    parser = argparse.ArgumentParser(description="Verify an AdderBoard benchmark submission")
    parser.add_argument("submission", help="Path to submission .py file")
    parser.add_argument("--num-tests", type=int, default=10000, help="Number of random tests")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--progress-every", type=int, default=1000, help="Print progress every N cases"
    )
    parser.add_argument(
        "--max-failures", type=int, default=20, help="Maximum number of failures to print"
    )
    args = parser.parse_args()

    mod = load_submission(args.submission)
    result = run_test(
        mod,
        num_tests=args.num_tests,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    _print_report(result, max_failures=args.max_failures)


if __name__ == "__main__":
    main()
