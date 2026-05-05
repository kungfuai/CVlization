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
        self.assertEqual(parts, ["P1", "P2", "P1"])

    def test_count_notes(self):
        mxc2 = "N C4 quarter\n+N E4 quarter\nR quarter\nN G4 half"
        self.assertEqual(_count_notes(mxc2), 3)


class TestLCS(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(_lcs_length(["C4", "D4", "E4"], ["C4", "D4", "E4"]), 3)

    def test_subset(self):
        self.assertEqual(_lcs_length(["C4", "D4", "E4", "F4"], ["C4", "E4", "F4"]), 3)

    def test_no_overlap(self):
        self.assertEqual(_lcs_length(["C4", "D4"], ["F5", "G5"]), 0)

    def test_empty(self):
        self.assertEqual(_lcs_length([], ["C4"]), 0)


class TestCombinedReward(unittest.TestCase):

    def _make_mxc2(self, pitches):
        lines = ["P1 Voice", "---", "P1", "M 1 key=0 time=4/4 clef=G2"]
        for p in pitches:
            lines.append(f"N {p} quarter su")
        return "\n".join(lines)

    def test_perfect_match_high_reward(self):
        ref = self._make_mxc2(["C4", "D4", "E4", "F4"])
        pred = self._make_mxc2(["C4", "D4", "E4", "F4"])
        scores = combined_reward([pred], [ref])
        # 100% similarity → score = 4.0 - 1.0 = 3.0
        self.assertAlmostEqual(scores[0], 3.0, places=1)

    def test_zero_match_low_reward(self):
        ref = self._make_mxc2(["C4", "D4", "E4", "F4"])
        pred = self._make_mxc2(["A5", "B5", "G3", "F3"])
        scores = combined_reward([pred], [ref])
        # 0% similarity → score = 0 - 1.0 = -1.0
        self.assertAlmostEqual(scores[0], -1.0, places=1)

    def test_correct_scores_higher_than_wrong(self):
        ref = self._make_mxc2(["C4", "D4", "E4", "F4"])
        pred_good = self._make_mxc2(["C4", "D4", "E4", "F4"])
        pred_bad = self._make_mxc2(["A5", "B5", "G3", "F3"])
        scores = combined_reward([pred_good, pred_bad], [ref, ref])
        self.assertGreater(scores[0], scores[1])

    def test_empty_prediction_negative(self):
        ref = self._make_mxc2(["C4", "D4"])
        scores = combined_reward([""], [ref])
        self.assertLess(scores[0], 0)

    def test_partial_match_intermediate(self):
        ref = self._make_mxc2(["C4", "D4", "E4", "F4"])
        pred = self._make_mxc2(["C4", "D4", "A5", "B5"])  # 50% match
        scores = combined_reward([pred], [ref])
        # Should be between perfect (3.0) and zero (-1.0)
        self.assertGreater(scores[0], -1.0)
        self.assertLess(scores[0], 3.0)

    def test_reward_correlates_with_eval_metric(self):
        """Verify reward ordering matches what SequenceMatcher.ratio() would give."""
        from difflib import SequenceMatcher
        ref = self._make_mxc2(["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"])
        preds = [
            self._make_mxc2(["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]),  # 100%
            self._make_mxc2(["C4", "D4", "E4", "F4", "X", "X", "X", "X"]),  # ~50%
            self._make_mxc2(["X", "X", "X", "X", "X", "X", "X", "X"]),  # 0%
        ]
        rewards = combined_reward(preds, [ref] * 3)
        # Rewards should be strictly decreasing
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])

        # Verify they match SequenceMatcher ordering
        ref_pitches = _extract_pitches(ref)
        sims = [SequenceMatcher(None, _extract_pitches(p), ref_pitches).ratio() for p in preds]
        # Same ordering
        self.assertGreater(sims[0], sims[1])
        self.assertGreater(sims[1], sims[2])


if __name__ == "__main__":
    unittest.main()
