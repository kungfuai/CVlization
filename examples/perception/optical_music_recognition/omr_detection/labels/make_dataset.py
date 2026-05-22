"""Build a detection-format HF dataset from rendered synthetic + openscore.

Reads per-page SVGs (already produced by `datasets/omr/pipeline.py`),
calls `extract_bboxes.extract_layout`, and emits a HF Dataset row per page:

    {
        "image":     PIL.Image,             # rendered PNG
        "score_id":  str,
        "page":      int,
        "n_pages":   int,
        "key":       int,                   # from source MusicXML, page-level
        "bboxes": {
            "systems":      [[x, y, w, h], ...],
            "staves":       [[sys_i, staff_i, x, y, w, h], ...],
            "barlines":     [[sys_i, x, y, h], ...],
            "key_sigs":     [[sys_i, staff_i, x, y, w, h, key_value], ...],
            "clefs":        [[sys_i, staff_i, x, y, w, h, clef_type], ...],
            "bar_numbers":  [[sys_i, measure_number, x, y], ...],
        },
    }

Status: STUB. Implement once `extract_bboxes` is fleshed out.

Typical usage (when implemented):
    python make_dataset.py --corpus l7a --output zzsi/synthetic-detection
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="l7a",
                   help="l7a | l9 | openscore_lieder")
    p.add_argument("--push-to-hub", default=None,
                   help="HF repo (e.g. zzsi/synthetic-detection)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap rows for quick iteration")
    args = p.parse_args()
    raise NotImplementedError("Implement bbox extraction first, then this.")


if __name__ == "__main__":
    main()
