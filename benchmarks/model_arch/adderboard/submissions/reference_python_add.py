"""
Reference submission for sanity-checking the benchmark harness.

This intentionally uses Python arithmetic and is NOT a valid transformer submission
for leaderboard purposes. It exists only to verify benchmark wiring.
"""


def build_model():
    return object(), {
        "name": "Reference Python Add",
        "author": "CVlization",
        "params": 0,
        "architecture": "non-transformer reference",
        "tricks": ["python integer arithmetic"],
    }


def add(model, a: int, b: int) -> int:
    del model
    return a + b
