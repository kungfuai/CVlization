"""SVG → layout bboxes.

For one LilyPond-rendered SVG of a page, extract:
  - `systems`:  [(x, y, w, h), ...]  ordered top→bottom
  - `staves`:   [(system_idx, staff_idx_in_system, x, y, w, h), ...]
  - `barlines`: [(system_idx, x_position, y, h), ...]
  - `key_sigs`: [(system_idx, staff_idx, x, y, w, h, key_value?), ...]
  - `clefs`:    [(system_idx, staff_idx, x, y, w, h, clef_type?), ...]
  - `bar_numbers`: [(system_idx, measure_number, x, y), ...]  (reused from pipeline.py)

Key signature *values* and clef *types* are derived later from the source
MusicXML — this module just emits coordinates and lets the caller attach
labels.

Status: STUB — implement against LilyPond SVG path patterns.

Strategy:
  1. Reuse `datasets/omr/pipeline.py` SVG bar-number extraction to get
     `bar_numbers` directly.
  2. Group bar numbers by y-coordinate band → systems.
  3. Detect staff lines as long horizontal `<path>` elements (LilyPond
     emits 5-line staves as 5 parallel horizontals per staff). Group them
     into staves.
  4. Detect barlines as short vertical `<path>` elements at consistent x
     positions across a system's staves.
  5. Optionally: detect sharp/flat/clef glyphs by path-data fingerprints
     known to LilyPond's Emmentaler font.

Reference data:
  - `datasets/omr/pipeline.py::extract_bar_nums_from_svg` — proven SVG
    bar-number extraction.
  - LilyPond's font glyph IDs are stable across versions; SVG path `d`
    attributes for `accidentals.sharp`, `accidentals.flat`, etc. are
    recognisable.

Public API (planned):
  extract_layout(svg_path: Path) -> dict

  where dict keys are: systems, staves, barlines, key_sigs, clefs, bar_numbers.
"""

import sys
from pathlib import Path

# Add the parent datasets package so we can reuse the existing SVG infra.
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from datasets.omr.pipeline import (  # noqa: E402
    extract_bar_nums_from_svg,
)


def extract_layout(svg_path: Path) -> dict:
    """Top-level entrypoint. Currently emits only bar numbers via the
    existing pipeline. Extend as detection-label work proceeds."""
    bar_nums = extract_bar_nums_from_svg(svg_path)  # list of {number, x, y}
    return {
        "bar_numbers": bar_nums,
        "systems": [],   # TODO
        "staves": [],    # TODO
        "barlines": [],  # TODO
        "key_sigs": [],  # TODO
        "clefs": [],     # TODO
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("svg")
    args = p.parse_args()
    import json
    print(json.dumps(extract_layout(Path(args.svg)), indent=2, default=str))
