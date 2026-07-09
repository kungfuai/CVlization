# Stage-1 KD Trainer — Plan

The Stage-1 training run: train the OmniAvatar-1.3B causal student to reproduce
the SoulX-FlashTalk teacher's talking-avatar videos, via flow-matching
regression on the teacher's latents.

This is the milestone after the KD dataset (now complete: 100 items of
portrait + audio + SoulX target video).

## Approach: adapt Self-Forcing's ODE trainer (do NOT extend e5_smoke.py)

`e5_smoke.py` did plain MSE on random tensors — it validated *plumbing*
(gradients, no NaN, memory), not a real training step. A correct Stage-1 run
needs the flow-matching noise schedule, OmniAvatar's input construction, and
the conditioning path.

Self-Forcing already ships this for Wan text-to-video:
- `model/ode_regression.py` — `ODERegression`: timestep sampling, `add_noise`,
  the regression loss (`generator_loss(ode_latent, conditional_dict)`).
- `trainer/ode.py` — training loop + FSDP + `ODERegressionLMDBDataset`.
- `scripts/generate_ode_pairs.py` / `create_lmdb_iterative.py` — data tooling.

Our Stage-1 trainer = this scaffold + audio conditioning + our student.

## Sub-tasks (each independently verifiable)

### S1 — Video → latent encode pass
- Load OmniAvatar's `WanVideoVAE` (Wan2.1 VAE; weights in
  `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`). It needs the latent `scale` constants.
- For each `soulx_targets/<id>.mp4`: read frames → tensor [3,F,H,W] in
  [-1,1] → `vae.encode` → latent [16,F_lat,H/8,W/8].
- Also encode the ref portrait → ref-image latent (for the conditioning).
- Cache as `<id>.pt`.
- Verify: shapes, value ranges; round-trip decode one and eyeball.
- Risk: VAE scale constants; the encode wrapper's tiled-mode args.

### S2 — Data format: pack into the trainer's expected layout
- `ODERegressionLMDBDataset` expects `ode_latent` (the teacher denoising
  trajectory) keyed in LMDB. Our SoulX targets are *final* latents, not
  multi-step trajectories.
- Decision: simplest correct form is single-point regression — treat the
  SoulX latent as the clean x0; the trainer samples a random timestep, noises
  it, the student predicts the flow/velocity. This is standard diffusion
  training on teacher data. (A full multi-step trajectory is only needed if we
  later want exact ODE-trajectory matching.)
- Build `pack_lmdb.py`: write each item's
  {video_latent, ref_latent, audio_emb, text_context} to LMDB.

### S3 — Audio path into the conditioning
- `audio_emb`: run the wav2vec encoder OmniAvatar uses
  (`facebook/wav2vec2-base-960h`) on each audio clip → features.
- Confirm the time-rate: `T_aud_packed = (T_audio_frames + 3) / 4` must equal
  `F_latent` (see ARCH_COMPARISON.md / EXPERIMENT_QUEUE.md A1).
- Precompute and cache per item (cheap; do it in S2's pack step).

### S4 — conditional_dict construction  [SOLVED — see findings below]
OmniAvatar's 33-channel DiT input, reverse-engineered from
`scripts/inference.py` L266-275:
```
x[0:16]   noisy video latent      (the 16-ch thing being denoised)
x[16:32]  ref-portrait latent     VAE-encode portrait -> [16,1,h,w], tiled to T
x[32:33]  mask                    0 at frame 0 (anchor), 1 elsewhere
```
i.e. `y = cat([ref_latent.repeat(1,1,T,1,1), mask], dim=ch)` -> 17ch;
`dit_input = cat([noisy_latent, y], dim=ch)` -> 33ch. Flow-matching noise is
applied ONLY to x[0:16]; ref + mask are clean conditioning.

Audio (`scripts/inference.py` L236-256): librosa.load(16k) ->
Wav2Vec2FeatureExtractor -> OmniAvatar's custom `Wav2VecModel` with
`output_hidden_states=True`; concatenate `last_hidden_state` + all 13 hidden
layers -> [T_audio, 10752]. T_audio = ceil(seconds * fps=25).

Text context: **smoke uses ZERO text context** — the dominant Stage-1 signal
is audio + ref + video reconstruction, and skipping it avoids wiring the
~11B umt5-xxl encoder. Real T5 conditioning is a v0 addition.

### S5 — Wire OmniAudioCausalWanModel into trainer/ode.py
- Replace Self-Forcing's `WanDiffusionWrapper` student with our
  `OmniAudioCausalWanModel` (loaded via `load_omni_into_causal_adapter`).
- Confirm FSDP wraps the subclass + the new audio modules cleanly.
- Thread `audio_emb` through `ODERegression.generator_loss`.

### S6 — First real Stage-1 run
- Smoke config: 100 items, ~200-500 steps, `disable_wandb`.
- Watch: does the loss decrease on REAL targets (vs E5's random)?
- Decode a student sample mid-training → eyeball: coherent talking face?
- This is the gate: recipe sound → scale to 2k/32k; broken → debug.

## Hardware
- Training fits 40 GB easily (single 1.3B student; cf. E5 = 5.4 GB peak).
- Use acasia (free GPUs) or drifter (Blackwell). Single GPU is enough for the
  100-item smoke.

## Effort estimate
~3-5 focused days to a clean Stage-1 loss curve. S4 (input construction) and
S6 (debugging the first run) dominate.

## Status
- [x] S1 — video→latent encode pass (100/100, encode_targets.py)
- [x] S2/S3 — per-item .pt (no LMDB needed at smoke scale); audio precomputed
- [x] (folded into S2: precompute_audio.py, 100/100)
- [x] S4 — 33-ch input solved (see findings above)
- [x] S5 — train_stage1.py (lean standalone trainer, reuses omni_causal_adapter)
- [x] S6 — trainer runs stably; eval metric added.
      HONEST RESULT: 1000-step run shows NO learning. Fixed-timestep eval
      0.516 -> 0.575 (+11.5%, slightly WORSE). The raw per-step loss is pure
      timestep noise. Trainer is correct mechanically (no NaN, ~9GB, 7s/step)
      but the recipe as configured does not improve the student.

## S6 result — no learning in 1000 steps. Candidate causes (untested):
- Too few steps / lr too low — real distillation is 6-8k+ steps; lr 1e-4 on
  a pretrained model is conservative. 1000 steps may simply be too short.
- Zero text context — OmniAvatar was trained WITH text conditioning; feeding
  zeros pushes the text cross-attn off-distribution. May need real T5.
- Zero-init audio_cond_projs barely move at lr 1e-4 in 1000 steps.
- Velocity parameterization verified correct (cross-checked vs SoulX
  pipeline: raw model output = noise - x0 = our v_target).

Next: debug the training (lr sweep, longer run, real text context, check
audio-module weight movement). This is open-ended ML debugging — the "S6
dominates" risk in the effort estimate. Recommend a deliberate diagnostic
pass rather than blind longer runs.
