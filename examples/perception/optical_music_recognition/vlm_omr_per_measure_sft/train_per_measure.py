#!/usr/bin/env python3
"""Per-measure SFT, extending safckylj at the per-measure scale.

STUB. Wire up once the per-measure dataset builder is verified.

Plan:
  1. Load JSONL from `--data` (records: image path + measure_mxc2).
  2. Build an HF Dataset of (PIL.Image, str) pairs.
  3. Init `FastVisionModel.from_pretrained(--vlm-ckpt)` (continue from
     safckylj). Apply LoRA on the language model + vision tower as in
     `vlm_omr_sft/train.py`.
  4. Trainer = SFTTrainer with INSTRUCTION_MEASURE = "Transcribe this
     single measure of sheet music to MXC2."
  5. max_seq_length tighter than the page model (256-512 should suffice).
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="dir from labels/build_per_measure_dataset.py")
    p.add_argument("--vlm-ckpt",
                   default="/vlm_sft/outputs/safckylj/final_model",
                   help="base VLM checkpoint to SFT off")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()
    raise NotImplementedError(
        "Build + verify the per-measure dataset first, then wire trainer.")


if __name__ == "__main__":
    main()
