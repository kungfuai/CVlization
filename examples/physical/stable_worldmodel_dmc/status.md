# stable-worldmodel CVL Status (2026-02-26)

## Question
Does `lejepa_epoch_50_object.ckpt` depend on a legacy top-level `module` namespace not present in released upstream `stable-worldmodel` code?

## Findings
1. The uploaded object checkpoint pickle references `module.*` globals.
   - Checked by reading `data.pkl` inside:
     - `/home/zsi/.cache/cvlization/stable-worldmodel/assets/models/swm-dmc-cheetah/lejepa_epoch_50_object.ckpt`
   - Result included:
     - `has_module True`
     - module list contains `module` (plus `jepa`, `torch`, `transformers`, etc.).

2. Released upstream tags do not contain a `module.py` file.
   - Repo checked: `/tmp/cvl/stable-worldmodel`
   - Tags checked: `v0.0.1a0`, `v0.0.1b0`, `v0.0.2b0`, `v0.0.1b1`, `0.0.2`, `0.0.1`, `0.0.4`, `0.0.5`
   - For each tag: `module.py_count=0`

3. With pinned upstream source in Docker (`v0.0.1b0`), strict world-model load still fails.
   - Command:
     - `./predict.sh "python run_eval.py --asset-dir /cvl-cache/stable-worldmodel/assets --skip-policy-rollout --validate-world-model-load --strict"`
   - Error:
     - `No module named 'module'`

4. The current pinned source API is older (`AutoCostModel(model_name, cache_dir=None)`), and we adapted loader invocation accordingly; failure still remains import-related (`module`).

## Conclusion
Yes, current evidence strongly supports that this checkpoint expects pickle-time class paths under top-level `module`, and that namespace is not available in released upstream tags we tested. So world-model object loading is blocked by checkpoint/code-version incompatibility, not by download corruption.

## Confidence
High.

## Resolution (2026-02-26)
Chose option 3: reconstruct architecture in `predict.py` from `lejepa_weights.ckpt`.  Model loads cleanly (303/363 keys, 0 strict-missing), synthetic forward pass verified.

---

# Full Pipeline Evaluation (Future Work — Doable, Significant Effort)

The original stable-worldmodel authors evaluate the world model via two approaches that
require the full goal-conditioned RL pipeline. Neither is implemented in this example yet,
but both are doable.

## 1. Goal-conditioned rollout success rate

**What it is:** The JEPA world model is used for model-based planning/RL.  A
goal-conditioned policy uses the world model's latent embeddings to navigate toward a goal
state recorded in the dataset.  Success rate = fraction of episodes where the agent reaches
the goal within a budget of steps.

**Why it is the right metric:** This directly tests whether the world model's learned
representations are useful for downstream control — the real purpose of JEPA.

**What is needed to implement:**
1. A goal-conditioned policy trained with JEPA embeddings (the expert locomotion policy in
   `swm-dmc-expert-policies` does NOT use world model embeddings — it is a plain SAC/PPO
   policy and cannot be used here).
2. The world model exposed via a compatible `AutoCostModel` interface.  Our `predict.py`
   Option-C reconstruction bypasses the broken `*_object.ckpt`, but the goal-conditioned
   policy expects to call the model through `stable_worldmodel.policy.AutoCostModel`.  A
   thin adapter wrapping our JEPA class in that interface would unblock this.
3. Integration with `swm.World.evaluate_from_dataset()` and the entry point
   `scripts/plan/eval_gcbc.py` in the stable-worldmodel repo.

**Estimated effort:** ~1–2 weeks.  The main blockers are (a) obtaining or training a
goal-conditioned policy checkpoint and (b) writing the `AutoCostModel` adapter.

## 2. Side-by-side predicted-vs-actual video from `evaluate_from_dataset()`

**What it is:** The world model guides a goal-conditioned policy rollout in the actual
MuJoCo environment.  At each step the environment renders the frame the agent actually
reached.  These rendered frames are stacked against the corresponding target frames from
the recorded dataset and saved as an MP4:

```
| Policy-rollout frame (t) |
| Dataset target frame (t) |  →  horizontally concatenated with goal frame
```

**Important:** this is NOT pixel-level prediction by the JEPA model.  JEPA (by design) has
no pixel decoder.  The "predicted" frames are real environment renders produced by a policy
that was guided by JEPA embeddings.  The comparison shows whether the policy (aided by the
world model) reproduces the recorded trajectory.

**What is needed:** identical to item 1 above — the goal-conditioned policy and the
`AutoCostModel` adapter.

**Estimated effort:** included in the ~1–2 weeks above; once the adapter is working, the
video generation is a one-liner via `world.evaluate_from_dataset(video_path=...)`.

## 3. Intermediate quality checks already implemented (no full pipeline needed)

These are implemented in `predict.py` today:
- **`--tsne`** — encode episode frames, run predictor, project to 2D with t-SNE; plots
  actual trajectory vs predicted positions to visualize embedding-space accuracy.
- **`--demo-video`** — nearest-neighbor retrieval: for each predicted next-frame embedding,
  find the closest frame in the dataset and show it alongside the actual next frame.  This
  is the standard JEPA evaluation method (used in I-JEPA, V-JEPA papers) and does not
  require a decoder or environment.

---

# Predictor Quality Analysis (2026-02-26)

## Finding: predictor is better than random but far from the copy baseline

Measured on the first 20 frames of `dmc/expert/cheetah/run.h5`:

| Embedding type | pred cos_sim | copy baseline | random |
|---|---|---|---|
| mean-pool patches | 0.197 | 0.9985 | 0.127 |
| CLS token | -0.040 | 0.147 | 0.062 |

Mean-pool is better than CLS.  Predictor is above random but far from the copy baseline.
The t-SNE plot (`--tsne`) confirms this: predicted embeddings (red stars) are scattered
broadly away from the actual trajectory (blue dots), with long connecting lines.

## Hypotheses investigated and ruled out

1. **Target/momentum encoder mismatch** — ruled out.  The checkpoint contains only
   `model.*` keys (encoder, predictor, action_encoder, pred_proj, projector).  There is no
   separate momentum or target encoder saved.  Our comparison (predictor output vs online
   encoder output) is correct.

2. **Wrong embedding type (CLS vs mean-pool)** — ruled out.  CLS performs worse, not
   better.

3. **Wrong embedding format (mean-pool vs full patch sequences)** — ruled out.  The public
   `stable-worldmodel` repo uses **PreJEPA** (patch sequences `(B, T, P, D)`).  Our
   checkpoint is **LeJEPA** — a different private model variant confirmed by weight shapes:
   `model.predictor.pos_embedding` is `(1, 3, 192)` (3 per-frame positions, not `3×P`
   patch positions).  LeJEPA's predictor is an action-conditioned DiT (adaLN-zero) that
   operates on per-frame mean-pooled vectors — our `encode_frames_batched()` is correct.

## Most likely cause: undertrained checkpoint

`config.yaml` sets `trainer.max_epochs: 100`.  The checkpoint is at **epoch 52** — halfway
through the intended training schedule.  JEPA-style self-supervised methods typically
require the full training run for the latent space to converge.  At epoch 52 the embedding
space is still organizing, which is why the t-SNE shows scattered predictions rather than
tight tracking of the actual trajectory.

## Recommended fix

Upload a fully-trained checkpoint (epoch 100) to `zzsi/swm-dmc-cheetah` and re-run the
quality checks.  No code changes are needed — `predict.py` will load it automatically.

