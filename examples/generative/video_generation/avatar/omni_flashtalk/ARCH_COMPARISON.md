# Architecture comparison — OmniAvatar audio path × Self-Forcing trainer

Notes from reading both codebases side-by-side on `acasia`. The goal is to
scope the audio-conditioning adapter we need to add to Self-Forcing's causal
Wan model so it can serve as the OmniAvatar-1.3B student in our distillation
recipe.

## OmniAvatar's audio path — concrete

Source: `OmniAvatar/OmniAvatar/models/wan_video_dit.py` (cloned to
`~/zz/OmniAvatar` on acasia) + `OmniAvatar/OmniAvatar/models/audio_pack.py`.

**Important**: OmniAvatar's audio conditioning is **not** cross-attention.
It's a per-block **additive conditioning**, lightweight and easy to port.

### Pipeline
```
raw audio
   └→ wav2vec2 (facebook/wav2vec2-base-960h, 768-dim per timestep)
       └→ multi-layer features concatenated → audio_input_dim = 10752 channels
           └→ AudioPack: rearrange + Linear(10752 → audio_hidden_size=32) + LayerNorm
               └→ audio_emb: shape (B, num_audio_blocks, 32)
                   └→ for each conditioned DiT block i, audio_cond_projs[i]: Linear(32 → dim)
```

### Where it's injected — `wan_video_dit.py` forward, lines ~388-405
```python
for layer_i, block in enumerate(self.blocks):
    if self.use_audio:
        if layer_i <= len(self.blocks) // 2 and layer_i > 1:   # blocks 2..N/2 inclusive
            au_idx = layer_i - 2
            audio_cond_tmp = patchify(audio_emb[:, au_idx].repeat(spatial))
            x = audio_cond_tmp + x                              # ADDITIVE, BEFORE the block
    x = block(x, context, t_mod, freqs)
```
- For a 30-layer Wan 1.3B (N=30), the audio condition is added before layers
  **2..15** — i.e., 14 blocks. So `len(audio_cond_projs) = num_layers // 2 - 1 = 14`.
- Each cond proj is `Linear(32, dim=1536)` ≈ 49K params; total audio path
  (AudioPack + 14 cond_projs) ~1M params. Trivially small on top of 1.3B.

### Stand-alone parts to lift
| Component | File | Lines | Params |
|---|---|---|---|
| `AudioPack` | `OmniAvatar/models/audio_pack.py` | full file | ~340K |
| `audio_cond_projs` ModuleList | `wan_video_dit.py` lines 311-318 | 8 lines | ~700K |
| Forward injection loop | `wan_video_dit.py` lines 388-405 | ~18 lines | n/a |
| Wav2vec wrapper | `OmniAvatar/models/wav2vec.py` | full file | (encoder weights) |
| Audio preprocessing | `OmniAvatar/utils/audio_preprocess.py` | full file | n/a |

The released `OmniAvatar/OmniAvatar-1.3B` checkpoint (339 MB) contains exactly
these audio components plus a LoRA on the Wan DiT — that is, a delta on top of
`Wan-AI/Wan2.1-T2V-1.3B`. The LoRA + audio modules together are the
"OmniAvatar" contribution.

### OmniAvatar inference regime (Stage-2 teacher behavior)
From `configs/inference_1.3B.yaml`:
- `num_steps: 50` — many-step (good for DMD teacher)
- `guidance_scale: 4.5` — standard CFG
- `seq_len: 200` (latent tokens per chunk)
- `overlap_frame: 13` — motion-frame conditioning for streaming
- `max_hw: 720` — 480p output (520×... or 720× something based on `image_sizes_720`)

So OmniAvatar-1.3B is **bidirectional and many-step at inference** — exactly
the Stage-2 teacher we want. Confirmed.

## Self-Forcing's training stack — concrete

Source: `~/zz/Self-Forcing` (cloned). 3349 stars, NeurIPS 2025 Spotlight.

### Models
| File | What |
|---|---|
| `model/base.py` | `SelfForcingModel` parent class |
| `model/ode_regression.py` | Stage-1: regress student onto pre-computed ODE pairs |
| `model/dmd.py` | Stage-2: DMD2 distillation (3 nets: gen + frozen real + trainable fake) |
| `model/causvid.py`, `model/gan.py`, `model/sid.py` | Alternative distillation losses |
| `wan/modules/causal_model.py` | **`CausalWanModel`** — causal Wan DiT with block-causal mask + KV cache. **This is what we extend.** |
| `wan/modules/model.py` | Standard (bidirectional) Wan model |

### Trainers
| File | What |
|---|---|
| `trainer/ode.py` | Stage-1 training loop |
| `trainer/distillation.py` | Stage-2 DMD training loop (analogue of Hallo's `distillation_fusion.py`) |
| `pipeline/self_forcing_training.py` | The AR self-rollout used inside DMD — block-by-block, KV cache, exit_flag gradient |
| `pipeline/causal_inference.py` | Inference-time AR loop |

### Configs (`configs/`)
- `self_forcing_dmd.yaml` — DMD recipe; teacher `Wan2.1-T2V-14B`, student `ode_init.pt`, 4-step `[1000,750,500,250]`, `num_frame_per_block: 3`, `dfake_gen_update_ratio: 5`. Almost identical structure to Hallo's `dmd_fusion_5B.yaml`.
- `self_forcing_sid.yaml` — SID variant.
- `default_config.yaml` — `causal: true`, `num_training_frames: 21`, etc.

### Critical observation
Self-Forcing's training stack is the **T2V-only ancestor of Hallo-Live's
training stack** — same architecture, same loss, same self-forcing pipeline,
minus the audio stream. Adding audio conditioning brings it back to parity
with what SoulX-FlashTalk (built on Self-Forcing) actually trained.

## The engineering gap — what we add

```
Self-Forcing's CausalWanModel  +  OmniAvatar's audio path  =  our student/critic
                                   (lifted verbatim from
                                    wan_video_dit.py)
```

Concretely, in `wan/modules/causal_model.py`:

1. **Add to `CausalWanModel.__init__`** (mirror OmniAvatar lines 307-318):
   ```python
   self.use_audio = args.get("use_audio", False)
   if self.use_audio:
       self.audio_proj = AudioPack(10752, [4,1,1], 32, layernorm=True)
       self.audio_cond_projs = nn.ModuleList([
           nn.Linear(32, dim) for _ in range(num_layers // 2 - 1)
       ])
   ```

2. **Add to `CausalWanModel.forward`** (mirror OmniAvatar lines 376, 384-405):
   - Accept `audio_emb` kwarg.
   - Project it once: `audio_emb = self.audio_proj(prepared_audio)`.
   - Concat per-block projections.
   - Inside the block loop, additively inject at layers `[2, N/2]`.

3. **Adapter on the data path**:
   - Add `OmniAvatar/models/wav2vec.py` and `utils/audio_preprocess.py` (or
     thin equivalents) as a module under `omni_flashtalk/trainer/audio/`.
   - Wire raw audio → wav2vec features → conditioning into the dataloader.

4. **State-dict mapping for warm-start**:
   - Load `OmniAvatar-1.3B` checkpoint (LoRA + audio modules) into the
     causal student. The LoRA targets the same Linear layers in the Wan DiT
     that exist in `CausalWanModel`. Need a small key-rename script to map
     OmniAvatar's parameter names to Self-Forcing's `CausalWanModel` names
     (likely `model.blocks.{i}.self_attn.{q,k,v,o}.weight` etc — verify by
     diffing state-dict keys).

5. **Stage-2 teacher path**:
   - The teacher is the *bidirectional* OmniAvatar (`wan/modules/model.py`-style),
     same weights as the student but with the standard Wan model (not causal).
   - Use `wan/modules/model.py` for the teacher; both load the same checkpoint.

## Open questions for Phase 0

1. **`num_audio_blocks` time dimension.** OmniAvatar's `AudioPack` uses
   `patch_size=[4,1,1]` — every 4 wav2vec timesteps become 1 audio block.
   For 25 fps video and 50 fps wav2vec, this gives 2 audio blocks per video
   frame; need to verify the rate alignment with Self-Forcing's
   `num_frame_per_block: 3`.

2. **Causal mask interaction with audio injection.** OmniAvatar is
   bidirectional → audio cond can see future audio. For causal streaming,
   should audio cond also be causally masked (only past audio visible per
   block), or is "all past + a small lookahead window" the right pattern?
   The released OmniAvatar code doesn't have a causal answer because it isn't
   causal. We need to pick one — most likely: per-block audio cond is computed
   from audio aligned with that video block plus a small lookahead window
   (analogous to `future_audio_frames` in Hallo-Live's
   `self_forcing_training_fusion.py`).

3. **LoRA vs full-finetune for student init.** OmniAvatar-1.3B was trained
   as a LoRA on Wan. For our distillation, do we merge the LoRA into base
   weights and full-finetune, or keep it as LoRA and only train the LoRA
   delta + audio modules + new causal-mask-related weights? Full-finetune is
   simpler trainer-side and probably better for distillation quality.

4. **Audio encoder for Stage-1 KD targets.** SoulX uses
   `chinese-wav2vec2-base`. If our student uses `wav2vec2-base-960h` and we
   try to imitate SoulX's outputs, the audio conditioning is being asked to
   map different feature distributions to the same target — an
   under-constrained problem. **Decision point:** either swap student's
   encoder to chinese-wav2vec2, or run SoulX inference using the English
   wav2vec encoder (probably degrades teacher quality), or use a multilingual
   encoder for both (extra training to re-condition both).
   *Tentative choice:* swap student encoder to `chinese-wav2vec2-base` and
   re-init the audio_proj input layer (10752 → 32 stays valid since base
   wav2vec2 features are the same dim). Cost: lose OmniAvatar's pretrained
   English audio_proj — re-trained during Stage-1 anyway.

5. **Validation that OmniAvatar-1.3B loads cleanly into the modified
   Self-Forcing CausalWanModel.** First concrete coding milestone — load the
   state dict, run a single forward pass with dummy audio_emb, confirm shapes.

## Downloads now on `acasia`

```
~/zz/omni_models/OmniAvatar-1.3B/        339M   (LoRA + audio modules)
~/zz/omni_models/Wan2.1-T2V-1.3B/         17G   (base model, T5, VAE, tokenizer)
~/zz/omni_models/wav2vec2-base-960h/     1.1G   (English wav2vec, OmniAvatar's encoder)
```

Already on disk from prior work:
```
~/zz/SoulX-FlashTalk/models/SoulX-FlashTalk-14B/      51G   (Stage-1 KD oracle)
~/zz/SoulX-FlashTalk/models/chinese-wav2vec2-base/   1.5G   (SoulX's audio encoder)
~/zz/Hallo-Live/model/                                58G   (Ovi, Wan, MMAudio, prompts)
```

Plenty of disk; everything we need for the recipe is local.

## Code locations (cloned)

- `~/zz/OmniAvatar/` — OmniAvatar repo
- `~/zz/Self-Forcing/` — Self-Forcing repo (the fork target)
- `~/zz/SoulX-FlashTalk/` — SoulX repo (for Stage-1 KD inference oracle)
- `~/zz/Hallo-Live/` — Hallo-Live repo (reference for trainer patterns, SyncNet)

## Phase 0 status

- [x] Both repos cloned and read.
- [x] OmniAvatar audio injection mechanism understood and documented.
- [x] Self-Forcing training stack mapped to Hallo-Live's analogue.
- [x] Student-init weights (OmniAvatar-1.3B + Wan-1.3B base + wav2vec) downloaded.
- [ ] Sketch the audio adapter as a patch on `wan/modules/causal_model.py`
      (next coding task — defer until dataset prep is unblocked).
- [ ] Validate state-dict load: OmniAvatar-1.3B → modified CausalWanModel.

Next: **prepare the Stage-1 KD dataset** — `(prompt, audio, ref_image)`
triplets to feed SoulX for trajectory generation. Plan in a separate doc.
