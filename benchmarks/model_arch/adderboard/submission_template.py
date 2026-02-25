"""
Submission template for the AdderBoard benchmark.

Your submission must define:

1. build_model() -> (model, metadata)
2. add(model, a: int, b: int) -> int
"""


def build_model():
    model = None  # Build your model here
    metadata = {
        "name": "My Adder",
        "author": "Your Name",
        "params": 0,  # Unique parameter count
        "architecture": "e.g. 1-layer decoder, d=4, 1 head",
        "tricks": ["e.g. tied embeddings", "factorized projections"],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    raise NotImplementedError("Implement model-based addition here.")
