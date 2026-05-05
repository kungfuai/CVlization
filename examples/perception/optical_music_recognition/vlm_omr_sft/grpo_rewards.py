"""GRPO reward functions for OMR.

Separated from train_grpo.py so they can be unit-tested without torch/unsloth.

The reward uses the EXACT same computation as the eval metric
(SequenceMatcher.ratio on pitched-only sequences) to ensure
reward and accuracy move in the same direction.
"""

import re
from difflib import SequenceMatcher


def _extract_pitches(text):
    """Extract pitched-note pitch tokens from MXC/MXC2 text."""
    pitches = []
    for line in text.split("\n"):
        # Match N <pitch> or +N <pitch> (not R = rests)
        m = re.match(r"[+]?N\s+(\S+)", line.strip())
        if m:
            pitches.append(m.group(1))
    return pitches


def _extract_parts(text):
    """Extract part IDs from MXC2 text."""
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("P") and not line.startswith("print"):
            parts.append(line.split()[0])
    return parts


def _count_notes(text):
    """Count note events in MXC2 text."""
    return sum(1 for line in text.split("\n")
               if re.match(r"\s*[+]?N\s", line))


def _lcs_length(a, b):
    """Longest common subsequence length (capped at 80 elements for speed)."""
    a = a[:80]
    b = b[:80]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def combined_reward(completions, ref_mxc2, **kwargs):
    """Reward that directly mirrors the eval metric.

    Uses SequenceMatcher.ratio() on pitched-only sequences — the EXACT
    same computation as pitched_only_similarity in eval_mxc.py.

    This ensures reward and accuracy move in the same direction.
    Previous versions used a custom LCS that was misaligned: reward
    went up while accuracy went down.
    """
    scores = []
    for pred, ref in zip(completions, ref_mxc2):
        pred_pitches = _extract_pitches(pred)
        ref_pitches = _extract_pitches(ref)

        if pred_pitches and ref_pitches:
            # Exact same computation as eval_mxc.py pitched_only_similarity
            sim = SequenceMatcher(None, pred_pitches, ref_pitches).ratio()
        else:
            sim = 0.0

        # Scale to [-1, +3] range: 0% → -1.0, 50% → +1.0, 100% → +3.0
        score = sim * 4.0 - 1.0
        scores.append(score)
    return scores
