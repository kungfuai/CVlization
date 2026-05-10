"""OMR reward function for Miles GRPO.

Miles calls `compute_score(response, label)` for each generated response.
Returns a float reward.

The reward uses SequenceMatcher.ratio() on pitched-only sequences —
the same computation as the eval metric (pitched_only_similarity).
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
    """Remove boilerplate from MusicXML."""
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    xml = re.sub(r'\s*<identification>.*?</identification>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    return xml.strip()


def _xml_to_mxc2_pitches(xml: str) -> list[str]:
    """Convert MusicXML to pitch list via MXC2.

    Falls back to extracting pitches directly from XML if mxc2 import fails.
    """
    try:
        import sys, os
        # Try importing mxc2 from the vlm_omr_sft directory
        sft_dir = os.path.join(os.path.dirname(__file__), "..", "vlm_omr_sft")
        if sft_dir not in sys.path:
            sys.path.insert(0, sft_dir)
        from mxc2 import xml_to_mxc2
        mxc2 = xml_to_mxc2(_strip_musicxml_header(xml), drop_beams=True)
        return _extract_pitches(mxc2)
    except Exception:
        # Fallback: extract pitches directly from XML
        pitches = []
        for m in re.finditer(r'<step>(\w)</step>.*?<octave>(\d)</octave>', xml, re.DOTALL):
            pitches.append(f"{m.group(1)}{m.group(2)}")
        return pitches


def compute_score(response: str, label: str) -> float:
    """Compute OMR reward for a single response.

    Args:
        response: Model-generated MXC2 text.
        label: Reference MusicXML (raw from dataset).

    Returns:
        Float reward in [-1, 3] range.
        -1.0 = no pitch match, +3.0 = perfect match.
    """
    pred_pitches = _extract_pitches(response)
    ref_pitches = _xml_to_mxc2_pitches(label)

    if not pred_pitches or not ref_pitches:
        return -1.0

    # Same computation as eval_mxc.py pitched_only_similarity
    sim = SequenceMatcher(None, pred_pitches, ref_pitches).ratio()

    # Scale to [-1, +3]
    return sim * 4.0 - 1.0
