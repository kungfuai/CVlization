# Experiment queue & default decisions

The plan has ~9 open questions we don't know the answers to. Rather than
debate them, **pick a default, mark it as a default (so we know what to
revisit), and let cheap experiments produce the data that disambiguates.**

## Defaults (so we can move)

Each entry: the question, the default, and why this default is *cheap to
revisit* if it turns out wrong. None of these defaults are commitments — they
are starting positions.

### Architecture

| # | Question | Default | Why this is the right default to start with |
|---|---|---|---|
| A1 | Audio time-rate (`AudioPack` patch size) | `[4,1,1]` — same as OmniAvatar | OmniAvatar-1.3B's pretrained audio modules were trained with this; changing it invalidates the warm start |
| A2 | Audio causal masking | **Fully causal** (only past audio per block, no lookahead) | Simpler; matches the streaming inference path. If lip-sync degrades, add a small `future_audio_frames` window later (Hallo-Live's analogue) |
| A3 | LoRA handling | **Merge LoRA into base weights, full-finetune** | Simplest trainer code; distillation quality is generally better with full-param freedom than constrained LoRA delta. Revisitable: keep base weights checkpointed pre-merge |
| A4 | Student audio encoder | **`wav2vec2-base-960h`** (OmniAvatar's English encoder) | Preserves OmniAvatar's pretrained audio modules. Implies SoulX-as-teacher is run with the same encoder (yes, possibly degrading SoulX quality on Chinese audio — acceptable for the v0) |
| A5 | State-dict load compatibility | **Assume the port works, verify empirically first** | Cheap to test (one forward pass). Don't pre-engineer renaming scripts before we know what mismatches actually exist |

### Dataset

| # | Question | Default | Why |
|---|---|---|---|
| D1 | KD target representation | **Latent only** (saved per-chunk after teacher denoising, before VAE decode) | 16× cheaper storage; loss operates in latent space anyway; pixel-aux can be added if MSE plateaus |
| D2 | Trajectories per item | **1** (single seed) | Single-trajectory is fine for a 2k-item smoke; multi-seed aug only adds value if student overfits |
| D3 | Source pairing | **Random** for v0 (prompt × audio × image pulled independently) | Diverse signal; semantic incoherence in some items is tolerable in supervised distillation. Curate later if signal is weak |
| D4 | Source corpora for smoke | **100 items**: 100 prompts from Hallo's 32k CSV × 100 random LibriSpeech-dev-clean 5s clips × ~10 rotated portraits from existing assets | Everything free, mostly already on disk, ~30 min of SoulX inference |

## Documented uncertainties (revisit triggers)

These are the **signals to watch** that should trigger revisiting a default:

| Signal | Triggers revisiting |
|---|---|
| Stage-1 training MSE plateaus high | D1 (add pixel-aux), A1 (audio rate), D3 (curate pairing) |
| Visible lip-sync drift in early checkpoints | A2 (add lookahead window), A4 (swap audio encoder) |
| State-dict load throws missing/unexpected keys | A5 (write rename script — but only the keys that actually mismatch) |
| Student diverges or generator_loss NaN | A3 (revert to LoRA-only), reduce lr |
| Stage-2 DMD teacher (bidirectional OmniAvatar) gives weak scores at random timesteps | Question whether OmniAvatar-1.3B's bidirectional mode is genuinely many-step trained (open question #4 in ARCH doc) |

## The first experiment

> **Goal**: confirm OmniAvatar-1.3B works end-to-end on `acasia` with its
> bundled sample, and that we understand what "the teacher (Stage-2) / the
> student init" actually does at inference time.

This is the cheapest probe that produces the most disambiguating signal.
Before we commit any time to porting code or building datasets, we should
know what OmniAvatar-1.3B's output looks like, how long inference takes on
40 GB, and what audio→video coupling we're starting from.

### What to run
```bash
# On acasia, in ~/zz/OmniAvatar
cd ~/zz/OmniAvatar
# (create venv, install OmniAvatar's requirements — different from Hallo's;
#  torch 2.4 per their README)
torchrun --standalone --nproc_per_node=1 scripts/inference.py \
    --config configs/inference_1.3B.yaml \
    --input_file examples/infer_samples.txt
```

The bundled `examples/infer_samples.txt` references `examples/images/0000.jpeg`
and `examples/audios/0000.MP3`. Output is a generated video.

### What to measure
1. **Does it run?** No crashes, no OOM on 1× A100-40GB.
2. **VRAM peak.** Confirms the 1.3B fits comfortably; gives us a baseline for
   the Stage-2 DMD memory budget (where teacher + student + critic are all
   1.3B).
3. **Wall time.** Per-chunk latency tells us how long Stage-1 KD data
   generation will take (we'd run SoulX, not OmniAvatar, for that — but
   the per-chunk-on-A100 number is a useful sanity reference).
4. **Output quality.** Eyeball the generated video. Is lip-sync visibly
   present? Is the avatar coherent across the 5-15s clip? This is the
   **floor** for our distilled student — we cannot beat OmniAvatar-1.3B's
   quality via distillation from it, only inherit it.

### What this tells us
- **A5 directly tested**: if OmniAvatar's own inference works, the state dict
  is internally consistent; subsequent ports are mechanical.
- **A1, A4 indirectly tested**: we see what the audio-cond + English-wav2vec
  combo produces at its native operating point.
- **Hardware budget validated**: confirms 1.3B at 480p fits a 40 GB card with
  room to spare (vs Hallo 5B which didn't).
- **Decision input for "do we even need SoulX as KD teacher"**: if
  OmniAvatar-1.3B output is already very good, Stage-1 KD from SoulX might
  be unnecessary and we could go straight to Stage-2 DMD (OmniAvatar
  many-step → OmniAvatar few-step). Cheaper recipe, simpler artifact.

### Failure modes & what they mean
- **Crashes during model load** → state-dict issue or bad config; investigate
  before any porting work.
- **OOM on 40 GB** → would be surprising for 1.3B; suggests their inference
  code lacks expected offload. Mitigation: set `num_persistent_param_in_dit`
  in the config (it's documented as their VRAM knob).
- **Garbage output (uncorrelated with audio)** → audio-cond modules not
  loaded or input format mismatch. Read inference.py.
- **Coherent video but no lip-sync** → audio path runs but conditioning is
  weak. Suggests A4 needs revisiting (encoder mismatch with what was
  trained), or audio_scale config is unset (it's `audio_scale:` blank in
  the default config).

### What this experiment is NOT
- Not a training experiment. No GPUs reserved for hours. Pure inference.
- Not testing our port. The port doesn't exist yet — we're testing the
  upstream code, then porting once we know it works.
- Not testing SoulX as teacher. That's the second experiment.

## Experiment queue & results

Each subsequent experiment is cheap (minutes-to-hours), each disambiguates
something specific. Pick the next one based on what the previous one revealed.

| # | Experiment | Status | What it told us |
|---|---|---|---|
| **E1** | OmniAvatar-1.3B inference on bundled sample | ✅ done | Runs cleanly, ~7 min wall time for 6s output, ~12 GB VRAM peak. Visible body + hand gestures (not just face). Validated A1, A4, A5 defaults. |
| **E2** | SoulX-FlashTalk-14B on same OmniAvatar inputs | ✅ done | Side-by-side: SoulX has more natural motion per user eyeball test. Confirms SoulX-as-teacher gives a meaningful quality gain over Omni-self-distill. |
| **E3** | Port OmniAvatar audio modules onto Self-Forcing's WanModel; load + forward | ✅ done | Adapter loads OmniAvatar's 634-key release cleanly into an SF-WanModel-derived class (`OmniAudioWanModel`). 0 unexpected keys, all audio + patch_embedding + LoRA-scaffold weights populated. Bidirectional forward pass runs (~3 GB VRAM, no errors). Code: `trainer/omni_adapter.py`. |
| **E3b** | Thread `audio_emb` through SF WanModel forward (per-block additive injection at layers [2..N/2]) | ✅ done | Forward with audio runs end-to-end. Audio contribution measurable (~2% relative signal magnitude with random audio). Time-rate alignment confirmed: `T_aud_packed = (T_audio_frames + 3) / 4` must equal `F_latent` (e.g. T_audio=81 ↔ F_lat=21). Code: `trainer/omni_adapter_v2.py`. |
| E3b2 (deferred) | Output match vs OmniAvatar's native `WanModel` on identical inputs | not yet | Would prove bit-equivalent port; non-trivial because SF's `_forward` differs from OmniAvatar's in internal plumbing (List[Tensor] vs single Tensor for x, etc). Defer unless training reveals a discrepancy that needs root-causing. |
| **E5** | Stage-1 KD smoke (random targets, 50 steps, MSE) | ✅ done | Training runs end-to-end. 1.3s/step, **5.4 GB peak** on 40 GB. Loss 1.65 → 1.04 (-37%), no NaN. Trainable subset = 177M (audio + LoRA + patch_embedding, 11% of total). Code: `trainer/e5_smoke.py`. |
| **E4** | Causal forward via Self-Forcing's `CausalWanModel` + audio adapter | ✅ done | Built a separate venv (torch 2.7.1+cu128, flash-attn cxx11abiTRUE). Adapter `OmniAudioCausalWanModel` subclasses `CausalWanModel`, ports `_forward_train` with audio injection and block-causal mask (flex_attention). Weights load cleanly via same loader pattern. Causal forward returns shape (B, 16, F, H, W). ~3 GB VRAM. Code: `trainer/omni_causal_adapter.py`, `trainer/e4_forward.py`. **Key fix learned**: `t` must be shape `[B, F]` not `[B]` (causal head expects per-frame timesteps for its modulation). |
| Phase 1a | Real Stage-1 dataset (per `DATASET_PLAN.md`): manifest builder + SoulX-trajectory generator | Produces actual KD targets to replace random tensors in E5 |
| Phase 2 (later) | Stage-2 DMD with 3× 1.3B nets (gen + critic + frozen teacher), self-forcing rollout | After E4 (causal mode) and Phase 1 (real dataset) are done |

## Decisions deferred (need data to answer)

These cannot be answered from documentation alone. They wait on experiments:

- Whether SoulX-as-Stage-1-teacher is worth the inference cost vs simpler
  "OmniAvatar→OmniAvatar self-distillation for causality + steps" (needs
  E1+E2 quality comparison).
- Whether OmniAvatar-1.3B bidirectional is a sufficient Stage-2 DMD teacher
  (needs E5 to start producing signal, plus a no-KD baseline).
- Whether 4-step → 1-step is feasible in one DMD pass or needs intermediate
  2-step (needs E5 plus a 4-step DMD run to first converge).

## Status

- [x] All defaults picked.
- [x] First experiment defined.
- [ ] **Run E1** — next concrete action.
