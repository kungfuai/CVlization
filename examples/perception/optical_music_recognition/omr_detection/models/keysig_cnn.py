"""Key-signature CNN — ported from vlm_omr_sft/train_keyclf_cnn.py.

Re-exports the model + helpers so omr_detection's pipeline can use the
already-trained classifier on cropped key-sig regions without depending
on the SFT folder's layout.

The trained weights live at:
  ../../vlm_omr_sft/outputs/keyclf_cnn_l7a/best.pt

98K params, hits 100% per-key on L7a dev. See
`reports/2026-05-21-respell-pipeline.md` for context.
"""

import sys
from pathlib import Path

_SFT_DIR = Path(__file__).resolve().parents[2] / "vlm_omr_sft"
sys.path.insert(0, str(_SFT_DIR))

# Direct re-export of the SFT-side definitions.
from train_keyclf_cnn import (  # noqa: E402, F401
    SmallKeyCNN,
    crop_keysig,
    fifths_to_label,
    label_to_fifths,
    _EVAL_TFM as EVAL_TFM,
    INPUT_W,
    INPUT_H,
    N_CLASSES,
)


__all__ = [
    "SmallKeyCNN", "crop_keysig", "fifths_to_label", "label_to_fifths",
    "EVAL_TFM", "INPUT_W", "INPUT_H", "N_CLASSES",
]
