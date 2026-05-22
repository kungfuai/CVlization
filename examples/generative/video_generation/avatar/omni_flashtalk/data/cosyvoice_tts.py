"""Generate one speech clip per manifest item with CosyVoice3.

Runs INSIDE the `cosyvoice3` docker image (/opt/CosyVoice on PYTHONPATH).
Model loaded once; all items synthesized in a loop. Resume-safe.

Output is resampled to 16 kHz mono — the rate SoulX-FlashTalk / OmniAvatar
expect for audio conditioning.

Smoke-stage limitation: all clips use the bundled default reference voice
(zero-shot mode). A future version would draw from a voice bank keyed off
each prompt's <AUDCAP> voice description for speaker diversity.

Usage (inside container):
    python cosyvoice_tts.py <manifest.jsonl> <output_dir>
"""
import json
import os
import sys

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_WAV = "/opt/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"  # transcript of the bundled prompt wav
TARGET_SR = 16000


def main():
    manifest_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    items = [json.loads(l) for l in open(manifest_path) if l.strip()]

    model = AutoModel(model_dir=MODEL_ID, fp16=False)
    src_sr = model.sample_rate
    print(f"CosyVoice3 loaded (sample_rate={src_sr})", flush=True)

    done = 0
    for it in items:
        out = os.path.join(out_dir, f"{it['id']}.wav")
        if os.path.exists(out):
            done += 1
            continue
        chunks = [
            r["tts_speech"]
            for r in model.inference_zero_shot(
                it["spoken_text"], PROMPT_TEXT, PROMPT_WAV, stream=False, speed=1.0
            )
        ]
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        if src_sr != TARGET_SR:
            audio = torchaudio.functional.resample(audio, src_sr, TARGET_SR)
        torchaudio.save(out, audio, TARGET_SR)
        done += 1
        dur = audio.shape[1] / TARGET_SR
        print(f"[{done}/{len(items)}] {it['id']} ({dur:.1f}s) -> {out}", flush=True)

    print(f"cosyvoice_tts done: {done}/{len(items)}")


if __name__ == "__main__":
    main()
