#!/usr/bin/env python3
"""Focused tests for duration/frame semantics in the LingBot-Video wrapper.

Tests the num_frames_from_duration() function, caption extraction, prompt
loading, and the argparse precedence logic (explicit --num-frames > JSON
duration > DEFAULT_NUM_FRAMES).

Run inside the Docker container or with predict.py on PYTHONPATH:
    python test_duration_frames.py
"""

import json
import os
import sys
import tempfile

# Import from predict.py in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import (
    DEFAULT_NUM_FRAMES,
    num_frames_from_duration,
    caption_from_sample,
    load_prompt_json,
)


def test_num_frames_from_duration():
    """Test upstream-compatible duration→num_frames derivation."""
    # 5s at 24fps → 120 raw frames → ((120-1)//4+1)*4+1 = 121
    assert num_frames_from_duration(5, 24) == 121, \
        f"5s@24fps should give 121, got {num_frames_from_duration(5, 24)}"

    # 3.4s at 24fps → int(81.6)=81 raw → ((81-1)//4+1)*4+1 = 85
    assert num_frames_from_duration(3.4, 24) == 85, \
        f"3.4s@24fps should give 85, got {num_frames_from_duration(3.4, 24)}"

    # 80/24 ≈ 3.333s at 24fps → 80 raw → ((80-1)//4+1)*4+1 = 81
    assert num_frames_from_duration(80 / 24, 24) == 81, \
        f"80/24s@24fps should give 81, got {num_frames_from_duration(80/24, 24)}"

    # 1s at 24fps → 24 raw frames → ((24-1)//4+1)*4+1 = 25
    assert num_frames_from_duration(1, 24) == 25, \
        f"1s@24fps should give 25, got {num_frames_from_duration(1, 24)}"

    # 0.5s at 24fps → 12 raw frames → ((12-1)//4+1)*4+1 = 13
    assert num_frames_from_duration(0.5, 24) == 13, \
        f"0.5s@24fps should give 13, got {num_frames_from_duration(0.5, 24)}"

    # Result must always be 4n+1
    for dur in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
        n = num_frames_from_duration(dur, 24)
        assert (n - 1) % 4 == 0, \
            f"num_frames_from_duration({dur}, 24) = {n} is not 4n+1"
        assert n >= 1, f"num_frames must be >= 1, got {n} for {dur}s"

    print("  num_frames_from_duration: PASS")


def test_default_num_frames():
    """Verify DEFAULT_NUM_FRAMES is 81 (4n+1)."""
    assert DEFAULT_NUM_FRAMES == 81
    assert (DEFAULT_NUM_FRAMES - 1) % 4 == 0
    print("  DEFAULT_NUM_FRAMES: PASS")


def test_caption_from_sample():
    """Test structured caption extraction."""
    # With 'caption' key (dict)
    sample = {"caption": {"description": "test"}, "duration": 5}
    result = caption_from_sample(sample)
    assert '"description":"test"' in result
    assert "duration" not in result

    # With 'caption' key (string)
    sample2 = {"caption": "plain text caption", "duration": 3}
    result2 = caption_from_sample(sample2)
    assert result2 == "plain text caption"

    # Without 'caption' key (filters runtime keys)
    sample3 = {"description": "test", "duration": 5, "fps": 24, "width": 832}
    result3 = caption_from_sample(sample3)
    parsed = json.loads(result3)
    assert "description" in parsed
    assert "duration" not in parsed
    assert "fps" not in parsed

    print("  caption_from_sample: PASS")


def test_load_prompt_json():
    """Test structured prompt JSON loading."""
    # With duration field
    data = {"caption": {"desc": "a test scene"}, "duration": 5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        caption, duration = load_prompt_json(f.name)
    os.unlink(f.name)
    assert duration == 5
    assert '"desc":"a test scene"' in caption

    # Without duration field
    data2 = {"caption": {"desc": "no duration"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data2, f)
        f.flush()
        caption2, duration2 = load_prompt_json(f.name)
    os.unlink(f.name)
    assert duration2 is None
    assert '"desc":"no duration"' in caption2

    print("  load_prompt_json: PASS")


def test_num_frames_precedence():
    """Test the 3-level precedence logic for resolving num_frames.

    This simulates the argparse resolution in main():
      1. Explicit --num-frames wins
      2. Otherwise derive from structured prompt duration
      3. Otherwise use DEFAULT_NUM_FRAMES
    """
    # Case 1: explicit --num-frames always wins
    explicit = 41
    duration = 5  # would give 121
    result = _resolve_num_frames(explicit_num_frames=explicit, duration=duration)
    assert result == 41, f"Explicit should win: expected 41, got {result}"

    # Case 2: JSON duration used when no explicit --num-frames
    result2 = _resolve_num_frames(explicit_num_frames=None, duration=5)
    assert result2 == 121, f"Duration 5s should give 121, got {result2}"

    # Case 3: DEFAULT when no explicit and no duration
    result3 = _resolve_num_frames(explicit_num_frames=None, duration=None)
    assert result3 == DEFAULT_NUM_FRAMES, \
        f"Default should be {DEFAULT_NUM_FRAMES}, got {result3}"

    # Case 4: t2i mode ignores duration
    result4 = _resolve_num_frames(explicit_num_frames=None, duration=5, mode="t2i")
    assert result4 == DEFAULT_NUM_FRAMES, \
        f"t2i should use default, got {result4}"

    print("  num_frames_precedence: PASS")


def _resolve_num_frames(
    explicit_num_frames=None,
    duration=None,
    mode="t2v",
    fps=24,
):
    """Replicate the precedence logic from predict.py main()."""
    if explicit_num_frames is not None:
        return explicit_num_frames
    elif duration is not None and mode != "t2i":
        return num_frames_from_duration(duration, fps)
    else:
        return DEFAULT_NUM_FRAMES


if __name__ == "__main__":
    print("Running duration/frame semantics tests...")
    test_default_num_frames()
    test_num_frames_from_duration()
    test_caption_from_sample()
    test_load_prompt_json()
    test_num_frames_precedence()
    print("\nAll tests passed.")
