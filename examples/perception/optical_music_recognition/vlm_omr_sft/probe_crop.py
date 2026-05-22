#!/usr/bin/env python3
"""Test whether cropping a broken +3 page into per-system bands changes Mode 1.

Hypothesis: the wrong key belief is page-level. If cropped to a single system
the model might re-perceive the key correctly (or might not).

Output: per-crop, count D# vs D occurrences in the transcription. The reference
key=+3 has F#,C#,G# but D is NATURAL. Mode 1 = model writes D# everywhere.
"""
import re
import os
import torch
from PIL import Image

CHECKPOINT = "outputs/safckylj/final_model"


def main():
    from datasets import load_dataset
    from unsloth import FastVisionModel
    from train import prepare_inference_inputs, INSTRUCTION_MXC2

    os.makedirs("probe_crop_out", exist_ok=True)
    model, processor = FastVisionModel.from_pretrained(CHECKPOINT, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    ds = load_dataset("zzsi/synthetic-scores", "level7a", split="dev")

    for label, idx in [("broken_+3", 9), ("perfect_+3", 4), ("broken_+1", 17)]:
        sample = ds[idx]
        sid = sample.get("score_id")
        img = sample["image"]
        w, h = img.size
        # 3 systems on L7a pages — crop the top 60% into 3 equal bands
        music_top, music_bot = int(h * 0.04), int(h * 0.55)
        band_h = (music_bot - music_top) // 3
        crops = [img.crop((0, music_top + i*band_h, w, music_top + (i+1)*band_h))
                 for i in range(3)]
        # Full-page baseline
        for name, im in [("full", img)] + [(f"sys{i+1}", c) for i, c in enumerate(crops)]:
            im.save(f"probe_crop_out/{sid}_{name}.png")
            inputs = prepare_inference_inputs(processor, im, INSTRUCTION_MXC2).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4096,
                                     use_cache=True, do_sample=False)
            pred = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            with open(f"probe_crop_out/{sid}_{name}.pred.txt", "w") as f:
                f.write(pred)
            # Count Mode-1 signature for +3 (D# vs D) or +1 (C# vs C)
            key_h = re.search(r"key=(-?\d+)", pred)
            key_pred = key_h.group(1) if key_h else "?"
            dsh = len(re.findall(r"\bD#\d", pred))
            dn = len(re.findall(r"\bD\d", pred))
            csh = len(re.findall(r"\bC#\d", pred))
            cn = len(re.findall(r"\bC\d", pred))
            print(f"  {label} {sid} {name:<6}: key_header={key_pred:>3}  "
                  f"D={dn} D#={dsh}  C={cn} C#={csh}  chars={len(pred)}")
        print()


if __name__ == "__main__":
    main()
