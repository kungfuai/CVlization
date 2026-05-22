"""S2/S3: precompute wav2vec audio embeddings, merge into latents/<id>.pt.

Mirrors OmniAvatar's audio extraction (scripts/inference.py L236-256):
librosa.load(16k) -> Wav2Vec2FeatureExtractor -> OmniAvatar's custom
Wav2VecModel with output_hidden_states; concatenate last_hidden_state + all
13 hidden layers -> [T_audio, 10752]. T_audio = ceil(seconds * fps=25).

Appends `audio_emb` to each existing latents/<id>.pt (which already holds
video_latent + ref_latent from encode_targets.py).

Run inside the OmniAvatar venv:
    python precompute_audio.py --data-dir ~/zz/omni_flashtalk_data \
        --wav2vec ~/zz/omni_models/wav2vec2-base-960h
"""
import argparse
import json
import math
import os
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/zz/OmniAvatar"))
from transformers import Wav2Vec2FeatureExtractor
from OmniAvatar.models.wav2vec import Wav2VecModel

SR = 16000
FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--wav2vec", required=True)
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    w2v_path = os.path.expanduser(args.wav2vec)
    device = "cuda"

    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(w2v_path)
    encoder = Wav2VecModel.from_pretrained(w2v_path, local_files_only=True).to(device)
    encoder.feature_extractor._freeze_parameters()
    encoder.eval()
    print("wav2vec loaded")

    items = [json.loads(l) for l in
             open(os.path.join(data_dir, "manifest_assets.jsonl")) if l.strip()]

    done = 0
    for it in items:
        pt_path = os.path.join(data_dir, "latents", f"{it['id']}.pt")
        if not os.path.exists(pt_path):
            print(f"  WARN {it['id']}: no latents .pt (run encode_targets first)")
            continue
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "audio_emb" in blob:
            done += 1
            continue

        audio, _ = librosa.load(os.path.join(data_dir, it["audio_path"]), sr=SR)
        input_values = np.squeeze(feat_ext(audio, sampling_rate=SR).input_values)
        input_values = torch.from_numpy(input_values).float().to(device).unsqueeze(0)
        audio_len = math.ceil(len(audio) / SR * FPS)

        with torch.no_grad():
            hs = encoder(input_values, seq_len=audio_len, output_hidden_states=True)
            emb = hs.last_hidden_state
            for mid in hs.hidden_states:
                emb = torch.cat((emb, mid), dim=-1)
        audio_emb = emb.squeeze(0).cpu()  # [audio_len, 10752]

        blob["audio_emb"] = audio_emb
        torch.save(blob, pt_path)
        done += 1
        vl = blob["video_latent"]
        print(f"[{done}/{len(items)}] {it['id']}: audio_emb {tuple(audio_emb.shape)}  "
              f"(video_latent T_lat={vl.shape[1]}; expect T_audio ~= 4*T_lat-3 "
              f"= {4*vl.shape[1]-3})", flush=True)

    print(f"precompute_audio done: {done}/{len(items)}")


if __name__ == "__main__":
    main()
