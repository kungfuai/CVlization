"""OMR reward function for Miles GRPO.

Miles calls `async def reward_fn(args, sample, **kwargs)` where:
  - args: training arguments namespace
  - sample: has .response (generated text), .label (reference XML), .prompt
  - returns a float reward

The reward uses SequenceMatcher.ratio() on pitched-only sequences —
the same computation as eval_mxc.py pitched_only_similarity.
"""

import re
from difflib import SequenceMatcher


def _extract_pitches(text: str) -> list[str]:
    """Extract pitched-note pitch tokens from MXC/MXC2 text."""
    pitches = []
    for line in text.split("\n"):
        m = re.match(r"[+]?N\s+(\S+)", line.strip())
        if m:
            pitches.append(m.group(1))
    return pitches


def _strip_musicxml_header(xml: str) -> str:
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    xml = re.sub(r'\s*<identification>.*?</identification>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    return xml.strip()


def _xml_to_pitches(xml: str) -> list[str]:
    """Extract pitch list from MusicXML.

    Tries MXC2 converter from vlm_omr_sft (mounted at /cvlization_repo);
    falls back to direct regex extraction.
    """
    try:
        import sys
        sft_dir = "/cvlization_repo/examples/perception/optical_music_recognition/vlm_omr_sft"
        if sft_dir not in sys.path:
            sys.path.insert(0, sft_dir)
        from mxc2 import xml_to_mxc2
        mxc2 = xml_to_mxc2(_strip_musicxml_header(xml), drop_beams=True)
        return _extract_pitches(mxc2)
    except Exception:
        pitches = []
        for note in re.finditer(r'<pitch>(.*?)</pitch>', xml, re.DOTALL):
            content = note.group(1)
            step_m = re.search(r'<step>(\w)</step>', content)
            oct_m = re.search(r'<octave>(\d)</octave>', content)
            alter_m = re.search(r'<alter>(-?\d+)</alter>', content)
            if step_m and oct_m:
                step = step_m.group(1)
                octave = oct_m.group(1)
                if alter_m:
                    a = int(alter_m.group(1))
                    if a > 0:
                        step += '#' * a
                    elif a < 0:
                        step += 'b' * (-a)
                pitches.append(f"{step}{octave}")
        return pitches


async def reward_fn(args, sample, **kwargs) -> float:
    """Async OMR reward for Miles GRPO.

    Returns SequenceMatcher.ratio scaled to [-1, +3].
    -1.0 = no match, +3.0 = perfect match.
    """
    response = sample.response if hasattr(sample, "response") else str(sample)
    label = sample.label if hasattr(sample, "label") else ""

    pred_pitches = _extract_pitches(response)
    ref_pitches = _xml_to_pitches(label)

    if not pred_pitches or not ref_pitches:
        return -1.0

    sim = SequenceMatcher(None, pred_pitches, ref_pitches).ratio()
    return sim * 4.0 - 1.0
