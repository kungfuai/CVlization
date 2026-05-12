#!/usr/bin/env python3
"""SFT training with Audiveris hints as in-context input.

Same as train.py but the prompt is augmented with the Audiveris MXC2 hint:

    "Transcribe this sheet music page to MXC2 (compact MusicXML).
     Audiveris draft (may have errors):
     <audiveris_mxc2>
     Corrected MXC2 transcription:"

If a sample has no Audiveris hint (extraction failed or no entry in the
hints file), the prompt is the standard image-only instruction —
teaching the model to handle both cases.

Usage:
    python train_with_hints.py --config config_hint_level9.yaml \
        --hints-train audiveris_hints_train.jsonl \
        --hints-dev   audiveris_hints_dev.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Reuse most of train.py — only override the conversation builder.
sys.path.insert(0, str(Path(__file__).parent))
import train as base_train
from train import INSTRUCTION_MXC2

INSTRUCTION_WITH_HINT = (
    "Transcribe this sheet music page to MXC2 (compact MusicXML). "
    "A classical OMR system (Audiveris) provides this draft transcription. "
    "It may have errors — especially in voice assignment, chord grouping, and "
    "pitch ordering — but it likely has correct measure boundaries, key/time "
    "signatures, clefs, and many correct pitch values. Use it as a hint and "
    "verify against the image.\n\n"
    "Audiveris draft:\n{hint}\n\n"
    "Corrected MXC2 transcription:"
)


def _load_hints(path: Path) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    hints = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("audiveris_failed", True) and r.get("audiveris_mxc2"):
                hints[r["score_id"]] = r["audiveris_mxc2"]
    return hints


def make_hinted_converter(orig_convert, hints, id_col, max_hint_chars=8000):
    """Wrap convert_to_conversation to inject hints into the user prompt."""
    def convert(sample):
        result = orig_convert(sample)
        sample_id = sample.get(id_col)
        hint = hints.get(sample_id) if sample_id else None
        if hint:
            hint_truncated = hint[:max_hint_chars]
            instruction = INSTRUCTION_WITH_HINT.format(hint=hint_truncated)
            # Replace the user-side text content with the hint-augmented instruction
            for msg in result["messages"]:
                if msg["role"] == "user":
                    for chunk in msg["content"]:
                        if chunk["type"] == "text":
                            chunk["text"] = instruction
                            break
                    break
        return result
    return convert


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hints-train", required=True,
                        help="JSONL mapping score_id -> audiveris_mxc2 for train split")
    parser.add_argument("--hints-dev", default=None,
                        help="Same for dev split (optional)")
    parser.add_argument("--hint-coverage", type=float, default=1.0,
                        help="Fraction of training samples to apply hint to (0..1). "
                             "Other samples use the standard image-only prompt, "
                             "teaching the model to handle both cases.")
    args = parser.parse_args()

    train_hints = _load_hints(Path(args.hints_train))
    dev_hints = _load_hints(Path(args.hints_dev)) if args.hints_dev else {}
    print(f"Loaded {len(train_hints)} train hints, {len(dev_hints)} dev hints")

    # Apply random hint dropout if requested
    if args.hint_coverage < 1.0:
        import random
        rng = random.Random(3407)
        keep = [k for k in train_hints if rng.random() < args.hint_coverage]
        train_hints = {k: train_hints[k] for k in keep}
        print(f"  Dropped {len(keep)} hints to hint_coverage={args.hint_coverage}")

    # Monkey-patch train.py's convert_to_conversation via a hook in main():
    # We re-implement what train.main() does but inject hints into the converter.
    # Simpler: import train.main and use a thread-local trick, OR just call into
    # the inner pieces. Easiest: invoke train.main with sys.argv adjusted, and
    # patch convert_to_conversation at the right point. But train.main is too
    # monolithic — we'd have to rewrite it.
    #
    # Pragmatic approach: store hints in a global that the patched convert_to_conversation
    # can read. We override convert_to_conversation at module level via dataset.map
    # callback wrapping.
    #
    # Cleanest: load the config, then call train_main() with hints injected via
    # a module-level global that train.py reads.
    base_train._INJECTED_HINTS_TRAIN = train_hints
    base_train._INJECTED_HINTS_DEV = dev_hints
    base_train._INJECTED_INSTRUCTION_WITH_HINT = INSTRUCTION_WITH_HINT

    # Re-construct argv for train.main
    sys.argv = [sys.argv[0], "--config", args.config]
    base_train.main()


if __name__ == "__main__":
    main()
