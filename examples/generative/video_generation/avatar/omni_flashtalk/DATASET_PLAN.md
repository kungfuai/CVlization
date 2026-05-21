# Stage-1 KD Dataset — design

The Stage-1 supervised KD step needs `(prompt, audio, reference_image)`
triplets to feed into the SoulX-FlashTalk-14B teacher. The teacher produces
video latent trajectories that the OmniAvatar-1.3B student learns to imitate.

This doc decides **what the dataset is, where it comes from, and how big it
needs to be** before any code is written.

## What an item looks like

Each item = one `(text_prompt, audio_clip, reference_image)` triplet that
SoulX consumes, producing one *output trajectory*. Per Stage-1, we save the
output's latent representation (not pixels) as the training target.

| Field | Format | Notes |
|---|---|---|
| `prompt` | string | scene/style description, ≤200 tokens |
| `audio` | 16 kHz mono WAV, ~5–15s | one utterance per item |
| `ref_image` | PNG/JPG of a face/person | aligned/cropped consistently |
| `output_latent` | tensor `[1, F_lat, C, H_lat, W_lat]` | the SoulX teacher output, saved per item |
| `output_audio_emb` | wav2vec features for the audio | precomputed once with the chosen encoder |

Stored in LMDB, mirroring Hallo-Live's `create_lmdb_fusion` layout so we can
reuse their dataloader code with minimal changes.

## Size — start small, scale on validated signal

- **Smoke**: 100 items. Verify the whole pipeline (generation → save → load
  → student forward → loss). Fits on one GPU in ~30 min of SoulX inference.
- **Phase-1 v0**: 2k items. Enough to see meaningful student convergence in
  Stage-1 over a few k training steps. ~20 hours of SoulX inference on a
  single A100-40GB (at ~35s per chunk + ~3-5 chunks per item).
- **Phase-1 v1**: 10k–32k items. Production scale, comparable to Hallo-Live's
  `synthetic_prompts_32k.csv` (32k prompts). 4–8 days of SoulX inference,
  parallelizable across all 8 GPUs on `acasia` (~12-24 hours wall).
- Push beyond 32k only if quality continues to improve.

The whole point of DMD2/self-forcing in Stage-2 is that it's dataset-free,
so the dataset cost is bounded — we only need enough Stage-1 data to give the
student a competent warm start.

## Source corpora — three concrete options ranked

### Option A (recommended) — bootstrap from existing assets we already have
Combine sources already on disk on `acasia`:

| Source | What | Count | Notes |
|---|---|---|---|
| Hallo-Live's `synthetic_prompts_32k.csv` | text prompts | 32k | already at `~/zz/Hallo-Live/prompts/data/` |
| SoulX bundled `examples/cantonese_16k.wav` + extracted public speech samples | audio | ~thousands | need source |
| LiveAvatar / SoulX bundled portrait samples | reference images | tens | need to expand |

The prompts are plentiful; audio + images are the bottleneck. To get to 2k
items quickly:
- **Audio**: pull from a public open dataset of short speech clips. Best
  free-licensed sources: **LibriSpeech-clean** (English, 1000h, CC-BY-4.0),
  **AISHELL-1** (Chinese, 178h, free for research). Random ~5-10s clips
  from these. Cheap, ethical, large.
- **Reference images**: pull from a public portrait dataset, e.g.
  **CelebA-HQ** or **FFHQ** thumbnails (Creative Commons-style licensing),
  or generate with a portrait-tuned diffusion model. ~100-1000 distinct
  portraits is enough — they get paired with many audio clips.

Pairing: random `(prompt, audio, ref_image)` from each pool. Even random
pairing gives the student diverse training signal.

### Option B — public talking-head video datasets, extract audio + first frame
Mature datasets in this domain:

| Dataset | Size | Audio? | License | Notes |
|---|---|---|---|---|
| **HDTF** | ~362 videos, ~16h | yes | research-only (paper-gated) | High-res talking heads |
| **VoxCeleb2** | 1M+ utterances, 6112 speakers | yes | CC-BY 4.0 with TOS | Standard for face recognition; lower quality |
| **CelebV-Text** | 70k clips, text-annotated | yes | research-only | Already includes text prompts — excellent fit |
| **AVSpeech** | 4.7k hours of clean speech-face video | yes | research-only | Google's dataset |

CelebV-Text is the closest fit (it ships `(text, video)` pairs; we'd derive
`(text, audio, first_frame)` from each). Costs: ~weeks of download/processing
for the full set; even a ~1% sample gives us ~700 items.

### Option C — synthesize audio from prompts via TTS
- Use Hallo-Live's 32k prompts directly.
- For each prompt, generate an audio clip with a TTS model (e.g.,
  Cosyvoice, F5-TTS — both open-weight, multilingual).
- Pair with random portrait from a portrait pool.
- **Upside**: prompt and audio are *coupled* (audio says what the prompt
  describes), which trains the model on aligned prompt+audio conditioning
  rather than random pairing.
- **Downside**: TTS quality bottleneck; synthetic-only training risks audio
  distribution mismatch at inference (real human speech).

## Recommended approach for `omni_flashtalk` v0

Hybrid, ordered by quick wins:

1. **Smoke set (100 items, Option A random pairing)** to validate the full
   data → SoulX → LMDB → student pipeline. Use:
   - 100 prompts from `synthetic_prompts_32k.csv`
   - 100 random ~5s clips from LibriSpeech `dev-clean` (a small, free subset)
   - 10-20 portraits from a small public set (CelebA-HQ thumbnails, or
     even the few bundled in LiveAvatar/SoulX examples, cycled)

2. **v0 set (2k items, Option A + Option C)**:
   - Half: random pairing as above.
   - Half: TTS-generated audio coupled to prompts (Cosyvoice or F5-TTS).
   - Same portrait pool, ~100 distinct portraits.

3. **v1 set (10-32k items, CelebV-Text + Option C bulk)** — only after v0
   gives signal that the recipe works.

## Implementation sketch — `data/sample_ode_data.py`

Modeled on Hallo-Live's `scripts/sample_ode_data.sh` flow but for our
audio-driven setting. Pseudo-flow:

```python
def main(manifest_path, output_lmdb, n_workers=8):
    items = read_manifest(manifest_path)  # one (prompt, audio_path, image_path) per line
    pipeline = load_soulx_pipeline(cpu_offload=True)
    audio_encoder = load_wav2vec2(chinese=True)  # SoulX's encoder

    lmdb_writer = open_lmdb(output_lmdb)
    for item in tqdm(items):
        prompt = item['prompt']
        audio_raw, sr = load_wav(item['audio_path'], target_sr=16000)
        ref_image = load_image(item['image_path'])

        # Run SoulX inference, save final latent trajectory
        with torch.no_grad():
            video_latents = pipeline.generate_latents(
                prompt=prompt,
                audio=audio_raw,
                ref_image=ref_image,
                return_latents=True,        # need to patch SoulX to expose this
            )

        # Pre-compute audio embedding the student will consume
        audio_emb = audio_encoder(audio_raw)

        lmdb_writer.put_item({
            'prompt': prompt,
            'audio_path': item['audio_path'],
            'image_path': item['image_path'],
            'video_latents': video_latents.cpu(),
            'audio_emb': audio_emb.cpu(),
        })
```

Two non-trivial bits:
- Patch SoulX's `generate_video.py` to expose the per-chunk latents (it
  currently only returns frames after VAE decode). Simple: add a flag and
  return `generated_list` (the latent list) before `save_video()`.
- Parallelize across the 8 A100s — each rank processes 1/8 of the manifest
  with its own SoulX pipeline instance. Embarrassingly parallel.

## Manifest format

Plain JSONL — one item per line:
```json
{"prompt": "A young woman is talking calmly.", "audio_path": "audio/lib_001.wav", "image_path": "portraits/celeba_042.png"}
```

Generate the manifest once with a small script (`data/build_manifest.py`)
that reads from the chosen sources and writes the JSONL. Easy to swap source
mixes without touching the trajectory generator.

## Open questions

1. **Latent or pixel target?** Latent is ~16× cheaper to store and the
   distillation loss operates in latent space anyway. Default: latent only.
   Add pixel-aux later if needed.
2. **One trajectory per audio, or N-shot augmentation?** Different seeds →
   different trajectories. Storing 1-2 trajectories per item is cheap and
   reduces the student's tendency to overfit to one seed's noise. Default: 1.
3. **Audio encoder selection.** Defer until ARCH_COMPARISON.md open question
   #4 is resolved. The chosen encoder also determines the manifest's audio
   format (mono / sample rate) and the precomputed `audio_emb` shape.
4. **Curate or sample?** Random pairing keeps the diversity high but
   produces some semantically incoherent items (e.g., "an angry man
   shouting" prompt + a quiet female utterance). Mild filtering (prompt
   gender ↔ speaker gender) is cheap and probably helps. Defer to v1.

## Phase 1a status

- [x] Decide what an item looks like.
- [x] Decide on size milestones (100 → 2k → 32k).
- [x] Pick source corpora (LibriSpeech + AISHELL + CelebA-HQ + TTS).
- [ ] Write `data/build_manifest.py` (next coding task).
- [ ] Patch SoulX's `generate_video.py` to expose latents.
- [ ] Write `data/sample_ode_data.py` and run smoke (100 items).
