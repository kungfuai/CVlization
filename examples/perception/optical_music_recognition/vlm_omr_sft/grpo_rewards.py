"""GRPO reward functions for OMR.

Separated from train_grpo.py so they can be unit-tested without torch/unsloth.
"""

import re


def _extract_parts(text):
    """Extract part IDs from MXC2 text."""
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("P") and not line.startswith("print"):
            parts.append(line.split()[0])
    return parts


def _extract_pitches(text):
    """Extract pitch sequence from MXC2 text."""
    pitches = []
    for line in text.split("\n"):
        m = re.match(r"[+]?N\s+(\S+)", line.strip())
        if m:
            pitches.append(m.group(1))
    return pitches


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
    """Single combined reward for GRPO.

    Components (weighted sum):
    - Pitch LCS similarity: 60% weight
    - Part count accuracy: 20% weight
    - Length control: 20% weight
    """
    scores = []
    for pred, ref in zip(completions, ref_mxc2):
        pred_pitches = _extract_pitches(pred)
        ref_pitches = _extract_pitches(ref)
        pred_parts = _extract_parts(pred)
        ref_parts = _extract_parts(ref)
        pred_notes = _count_notes(pred)
        ref_notes = _count_notes(ref)

        # Pitch LCS (60%)
        if pred_pitches and ref_pitches:
            lcs = _lcs_length(pred_pitches, ref_pitches)
            pitch_score = lcs / min(len(ref_pitches), 80)
        else:
            pitch_score = 0.0

        # Part count (20%)
        if ref_parts:
            part_score = 1.0 if len(pred_parts) == len(ref_parts) else 0.0
        else:
            part_score = 0.5

        # Length control (20%)
        if ref_notes > 0:
            coverage = pred_notes / ref_notes
            if 0.7 <= coverage <= 1.3:
                length_score = 1.0
            elif coverage > 2.0 or coverage < 0.3:
                length_score = 0.0
            else:
                length_score = 0.5
        else:
            length_score = 0.5

        score = (pitch_score * 0.6 + part_score * 0.2 + length_score * 0.2) * 4.0 - 1.0
        scores.append(score)
    return scores
