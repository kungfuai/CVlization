#!/usr/bin/env python3
"""Dump pred vs ref MXC2 for specific L7a dev samples (key-in-prompt model).

Writes inspect_out/<id>.pred.txt and <id>.ref.txt for hand inspection.
"""

import os
import re
import torch

CHECKPOINT = "outputs/cj882p8z/final_model"
# (index, label) — index into the level7a dev split
SAMPLES = [
    (9,  "broken"), (17, "broken"), (11, "broken"), (20, "broken"), (23, "broken"),
    (4,  "perfect"), (34, "perfect"), (13, "perfect"), (39, "perfect"),
]
OUT = "inspect_out"


def main():
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, INSTRUCTION_MXC2, strip_musicxml_header
    from eval_mxc import evaluate_pair
    from mxc2 import xml_to_mxc2

    os.makedirs(OUT, exist_ok=True)
    model, processor = FastVisionModel.from_pretrained(CHECKPOINT, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    ds = load_dataset("zzsi/synthetic-scores", "level7a", split="dev")

    for idx, label in SAMPLES:
        sample = ds[idx]
        sid = sample.get("score_id", f"dev_{idx}")
        m = re.search(r"<fifths>(-?\d+)</fifths>", sample["musicxml"] or "")
        key = int(m.group(1)) if m else None
        instruction = (f"{INSTRUCTION_MXC2} The key signature is key={key} "
                       f"(N = sharps if positive, flats if negative).")
        inputs = prepare_inference_inputs(processor, sample["image"], instruction).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8192, use_cache=True,
                                 do_sample=False)
        pred = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        ref = strip_musicxml_header(sample["musicxml"])
        try:
            ref = xml_to_mxc2(ref, drop_beams=True)
        except Exception:
            pass
        r = evaluate_pair(pred, ref, score_id=sid)
        with open(f"{OUT}/{sid}.{label}.pred.txt", "w") as f:
            f.write(pred)
        with open(f"{OUT}/{sid}.{label}.ref.txt", "w") as f:
            f.write(ref)
        print(f"{sid} [{label}] key={key} pitch={r.pitched_only_similarity:.1%} "
              f"pred_lines={pred.count(chr(10))+1} ref_lines={ref.count(chr(10))+1} "
              f"pred_chars={len(pred)} ref_chars={len(ref)}")


if __name__ == "__main__":
    main()
