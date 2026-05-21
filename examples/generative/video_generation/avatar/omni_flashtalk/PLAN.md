# omni_flashtalk — Plan

A small, real-time, causal streaming talking-avatar model produced by distilling
SoulX-FlashTalk-14B (teacher) into an OmniAvatar-1.3B-based student.

Working name `omni_flashtalk` (OmniAvatar arch + FlashTalk-lineage knowledge).
Will become a CVlization **training example** at
`examples/generative/video_generation/avatar/omni_flashtalk/`, alongside the
existing `soulx_flashtalk/` (inference of the SoulX teacher) and `live_avatar/`.

## Goal

A 1.3B-parameter audio-driven talking-avatar model that is:

- **Causal & streaming** — block-wise autoregressive with motion-frame
  conditioning (same streaming substrate as SoulX-FlashTalk).
- **Few-step** — 4 → 2 → 1 sampling steps via DMD distillation.
- **Small enough for consumer GPUs** — runs on a single 24 GB card with int8
  quanto + CPU offload, ideally real-time on RTX 4090/5090.

Success criterion (initial): the distilled model produces visually-coherent,
lip-synced 25 fps video on a single A100-40GB with `--cpu_offload`, with
quality measurably ahead of OmniAvatar-1.3B alone (since it inherits SoulX's
trajectory targets in Stage-1).

## Approach (hybrid, two stages)

The rejected alternatives and why are in the conversation log; in short:

- **Direct SoulX-as-DMD-teacher** has four real frictions (already-distilled
  teacher's score function is unreliable at random timesteps; causal teacher
  expects AR context that DMD's score call doesn't provide; audio-encoder
  mismatch chinese-wav2vec2 vs wav2vec2-base-960h; architectural reconciliation).
- **OmniAvatar-14B-teacher + OmniAvatar-1.3B-student** is clean but caps
  quality at OmniAvatar-14B (older than SoulX).
- **Hybrid (chosen)** uses SoulX only where it's strong (as an inference oracle
  for offline targets) and a same-family teacher where DMD is well-behaved.

### Stage 1 — Offline supervised KD from SoulX
- Run SoulX-FlashTalk-14B inference over a curated set of `(prompt, audio, ref_image)`
  inputs (~10–32k items). Save the resulting *latent* trajectories (one per
  generated chunk) to LMDB, mirroring Hallo-Live's `create_lmdb_fusion` format.
- Train an OmniAvatar-1.3B-arch *causal* student to regress those latents
  (MSE in latent space, optional pixel-space auxiliary later).
- This pass simultaneously (a) shrinks 14B → 1.3B and (b) imports SoulX's
  causal autoregressive behavior — both via supervised imitation.

### Stage 2 — DMD step reduction (same-family teacher)
- Teacher: OmniAvatar-1.3B in **bidirectional, many-step** mode (frozen).
- Student: causal 1.3B from Stage-1.
- Critic: fresh 1.3B copy.
- Self-Forcing autoregressive rollout + DMD2 loss (Hallo's distillation_fusion
  trainer, adapted for audio-driven not joint AV).
- Schedule: 4-step first → confirm stable → push to 2-step → 1-step.
- Optional HP-DMD with SyncNet reward (Hallo's `enable_rl_reward: true`).

### Stage 3 — Inference packaging
- `int8 optimum-quanto` (Ampere-friendly; SoulX-validated).
- `torch.compile` + flash-attn / cuDNN attention.
- Motion-frame chunked AR streaming (reuse OmniAvatar's streaming code).
- Dockerize as `cvl run omni_flashtalk predict`.

## Codebase choices

**Trainer**: fork **Self-Forcing** (https://github.com/guandeh17/Self-Forcing).
- It is Wan-T2V-based; SoulX-FlashTalk explicitly credits it as
  "the codebase we built upon". Closest fit for our target architecture.
- Add audio cross-attention modules (OmniAvatar-style injection at the
  configured DiT layers) and audio-encoder wiring.
- Add Hallo-Live-style `ode_fusion` and `dmd_fusion` trainers (or port their
  trainers directly — the DMD step + self-forcing rollout logic is reusable
  with the audio-driven wrapper swap).

**Student arch + init**: `OmniAvatar/OmniAvatar-1.3B` (LoRA + audio condition
weights, ~0.5 GB) on top of `Wan-AI/Wan2.1-T2V-1.3B` (~3 GB base).

**Teacher (Stage-1)**: `Soul-AILab/SoulX-FlashTalk-14B` (already downloaded on
`acasia` at `~/zz/SoulX-FlashTalk/models/SoulX-FlashTalk-14B/`, ~51 GB).

**Teacher (Stage-2)**: same `OmniAvatar-1.3B` checkpoint, just run in
bidirectional many-step mode (different attention mask + sampling schedule).
No second download.

**Audio encoder**: TBD. SoulX uses `chinese-wav2vec2-base`; OmniAvatar uses
`wav2vec2-base-960h`. For Stage-1 the student inherits whichever encoder
produces the (audio, ref_image) → video mapping we're imitating — likely
SoulX's, with student's audio cross-attn modules re-initialized to consume
SoulX's encoder features. (Open question — see Risks.)

## Repo layout

```
examples/generative/video_generation/avatar/omni_flashtalk/
├── PLAN.md                          # this file
├── README.md                        # (later) usage docs
├── Dockerfile                       # training + inference image
├── requirements.txt                 # frozen deps
├── build.sh                         # cvl build
├── train.sh                         # cvl train — orchestrates Stage 0/1/2
├── predict.sh                       # cvl predict — inference wrapper
├── example.yaml                     # CVlization metadata
├── configs/
│   ├── stage1_kd.yaml               # offline-KD config
│   └── stage2_dmd.yaml              # DMD config (4/2/1 step variants)
├── trainer/                         # forked Self-Forcing + audio adapter
│   └── (vendored or cloned at Docker build, like live_avatar does)
├── data/
│   └── sample_ode_data.py           # SoulX-trajectory-generation script
├── train.py                         # entry point (--stage {kd,dmd})
├── predict.py                       # inference wrapper
└── .gitignore
```

Code is developed locally in this directory and synced to `acasia` (where the
GPUs live) for execution. The Dockerfile pattern follows `live_avatar`/
`soulx_flashtalk`: clone upstream Self-Forcing at a pinned commit, COPY in our
adapter + trainer modifications, install requirements.

## Hardware & resource budget

All on `acasia` (8× A100-PCIE-40GB):

| Phase | Memory pressure | Wall time (estimate) |
|---|---|---|
| 0. Scaffold trainer | minimal | 3–5 days dev |
| 1a. SoulX trajectory generation | 1 GPU @ ~40 GB (SoulX inference, already validated) | ~2 days for 10k inputs @ ~35 s/chunk |
| 1b. Stage-1 student training | 8 GPUs @ ~5–10 GB each (single 1.3B network + Adam) | 3–5 days |
| 2. Stage-2 DMD | 8 GPUs @ ~10–15 GB each (3× 1.3B + Adam) | 3–5 days per step-count |
| 3. Inference packaging | minimal | 2–3 days |

Total: **~2–3 weeks** wall time for a first 1.3B causal 4-step model. Push to
2-step and 1-step adds ~1 week each.

Stage-2 fits comfortably on 40 GB because all three networks are 1.3B — the
problem that blocked Hallo-Live 5B DMD on 40 GB does not apply at this scale.

## Risks & open questions

1. **Audio-encoder reconciliation (Stage 1).** SoulX's trajectories were
   generated conditioned on Chinese-wav2vec2 features. The OmniAvatar-1.3B
   student's audio cross-attn was pretrained on English-wav2vec2 features.
   We must pick one encoder, accept that the other's pretrained audio modules
   are partially obsoleted, and let Stage-1 retrain them. Decision deferred
   until we read both repos' audio paths in detail.

2. **Self-Forcing audio-conditioning adapter.** Self-Forcing is T2V (no
   audio). Adding audio cross-attn cleanly is the main engineering task in
   Phase 0. Reference implementations: OmniAvatar (released), SoulX-FlashTalk
   (released, but built on Self-Forcing + InfiniteTalk — may have already
   solved this; read their `flash_talk_pipeline.py` for the pattern).

3. **Stage-1 quality ceiling.** The student inherits SoulX's quality only as
   far as supervised regression on its latents allows. Loss in pixel space
   (auxiliary) may be needed if latent-space MSE plateaus. Add only if needed.

4. **Stage-2 teacher = OmniAvatar-1.3B many-step.** Confirm OmniAvatar-1.3B is
   genuinely bidirectional and supports many-step sampling cleanly. If
   OmniAvatar shipped a few-step distilled variant, we need a different
   Stage-2 teacher (e.g., the base Wan2.1-T2V-1.3B + add audio cross-attn,
   sacrificing some quality for cleaner many-step behavior).

5. **HP-DMD reward stack.** Hallo-Live uses SyncNet for lip-sync reward. For
   audio-driven avatar this is essential. Adopt SyncNet from Hallo-Live (already
   downloaded with the Hallo training assets).

6. **Whether Stage-1 needs ODE-init-style trajectory regression at all.** Maybe
   a simpler "predict-the-final-latent" KD suffices given SoulX is already
   4-step. Try both; favor the simpler one if quality matches.

## Concrete first actions (Phase 0)

1. Clone OmniAvatar (https://github.com/Omni-Avatar/OmniAvatar) and Self-Forcing
   on `acasia`; read both architectures side-by-side. Document the audio
   cross-attn injection scheme in each and the audio-encoder plumbing.
2. Download `OmniAvatar/OmniAvatar-1.3B` (~0.5 GB) and `Wan-AI/Wan2.1-T2V-1.3B`
   (~3 GB) to `acasia`. Run OmniAvatar's inference end-to-end on a sample to
   verify the architecture loads and the audio path works.
3. Write the Stage-1 data-generation script (`data/sample_ode_data.py`) — runs
   SoulX inference over a prompts/audio/image manifest, saves latent
   trajectories to LMDB. Validate on 10 items first.
4. Sketch the audio-conditioning adapter to bolt onto Self-Forcing's DiT
   wrapper. This is the longest-pole engineering item in Phase 0.
5. Set up `configs/stage1_kd.yaml` with a tiny-dataset smoke target
   (~100 items, ~50 training steps) before scaling up.

## Notes

- Naming: `omni_flashtalk` is a working title. The artifact this produces is
  not affiliated with the SoulX or OmniAvatar teams — rename before any
  public release.
- The CVlization example will eventually expose both `train` (the full pipeline)
  and `predict` (inference with the resulting checkpoint) presets, with the
  distilled checkpoint hosted on HuggingFace under our own namespace once
  it works.
