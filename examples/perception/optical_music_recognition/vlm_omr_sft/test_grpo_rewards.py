"""Unit tests for GRPO reward functions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grpo_rewards import (
    _extract_pitches, _extract_parts, _count_notes,
    _lcs_length, combined_reward,
)


class TestExtractors(unittest.TestCase):

    def test_extract_pitches(self):
        mxc2 = "N C4 quarter su\nN D4 half\n+N E4 quarter\nR quarter\nN F4 eighth"
        pitches = _extract_pitches(mxc2)
        self.assertEqual(pitches, ["C4", "D4", "E4", "F4"])

    def test_extract_pitches_empty(self):
        self.assertEqual(_extract_pitches("M 1 key=0 time=4/4\nR whole"), [])

    def test_extract_parts(self):
        mxc2 = "header title\nP1 Voice\nP2 Piano\n---\nP1\nM 1\nN C4 quarter"
        parts = _extract_parts(mxc2)
        self.assertEqual(parts, ["P1", "P2", "P1"])  # declaration + usage

    def test_extract_parts_no_print_match(self):
        mxc2 = "print new-system\nP1 Voice"
        parts = _extract_parts(mxc2)
        self.assertEqual(parts, ["P1"])

    def test_count_notes(self):
        mxc2 = "N C4 quarter\n+N E4 quarter\nR quarter\nN G4 half"
        self.assertEqual(_count_notes(mxc2), 3)


class TestLCS(unittest.TestCase):

    def test_identical(self):
        seq = ["C4", "D4", "E4"]
        self.assertEqual(_lcs_length(seq, seq), 3)

    def test_subset(self):
        a = ["C4", "D4", "E4", "F4"]
        b = ["C4", "E4", "F4"]
        self.assertEqual(_lcs_length(a, b), 3)

    def test_no_overlap(self):
        a = ["C4", "D4"]
        b = ["F5", "G5"]
        self.assertEqual(_lcs_length(a, b), 0)

    def test_offset(self):
        # Same pitches but shifted — LCS should still find them
        a = ["X", "C4", "D4", "E4"]
        b = ["C4", "D4", "E4", "X"]
        self.assertEqual(_lcs_length(a, b), 3)

    def test_empty(self):
        self.assertEqual(_lcs_length([], ["C4"]), 0)
        self.assertEqual(_lcs_length(["C4"], []), 0)


class TestCombinedReward(unittest.TestCase):

    def _make_mxc2(self, parts, pitches, n_notes=None):
        """Build minimal MXC2 for testing."""
        lines = []
        for p in parts:
            lines.append(f"{p} Voice")
        lines.append("---")
        lines.append(parts[0] if parts else "P1")
        lines.append("M 1 key=0 time=4/4 clef=G2")
        for pitch in pitches:
            lines.append(f"N {pitch} quarter su")
        return "\n".join(lines)

    def test_perfect_match(self):
        ref = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4", "F4"])
        pred = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4", "F4"])
        scores = combined_reward([pred], [ref])
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 1.5)  # should be high reward

    def test_wrong_pitches(self):
        ref = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4", "F4"])
        pred = self._make_mxc2(["P1", "P2"], ["A5", "B5", "G3", "F3"])
        scores_wrong = combined_reward([pred], [ref])
        pred_right = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4", "F4"])
        scores_right = combined_reward([pred_right], [ref])
        # Wrong pitches should score lower than correct pitches
        self.assertGreater(scores_right[0], scores_wrong[0])

    def test_wrong_part_count(self):
        ref = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4"])
        pred = self._make_mxc2(["P1", "P2", "P3", "P4"], ["C4", "D4", "E4"])
        scores_wrong = combined_reward([pred], [ref])
        pred_right = self._make_mxc2(["P1", "P2"], ["C4", "D4", "E4"])
        scores_right = combined_reward([pred_right], [ref])
        self.assertGreater(scores_right[0], scores_wrong[0])

    def test_empty_prediction(self):
        ref = self._make_mxc2(["P1"], ["C4", "D4"])
        scores = combined_reward([""], [ref])
        self.assertLess(scores[0], 0)

    def test_overgeneration_penalty(self):
        ref = self._make_mxc2(["P1"], ["C4", "D4"])
        pred = self._make_mxc2(["P1"], ["C4"] * 20)  # 10x overgen
        scores = combined_reward([pred], [ref])
        pred_right = self._make_mxc2(["P1"], ["C4", "D4"])
        scores_right = combined_reward([pred_right], [ref])
        self.assertGreater(scores_right[0], scores[0])

    def test_batch(self):
        ref1 = self._make_mxc2(["P1"], ["C4", "D4"])
        ref2 = self._make_mxc2(["P1"], ["E4", "F4"])
        pred1 = self._make_mxc2(["P1"], ["C4", "D4"])  # perfect
        pred2 = self._make_mxc2(["P1"], ["A5", "B5"])  # wrong
        scores = combined_reward([pred1, pred2], [ref1, ref2])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
