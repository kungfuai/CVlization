# stable-worldmodel DMControl

World model loading (LeJEPA) + expert policy rollout for the DeepMind Control Suite cheetah task, using the published HuggingFace assets from [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## Quick Start

```bash
cd examples/physical/stable_worldmodel_dmc
./build.sh
./predict.sh
```

This will:
1. Download HuggingFace assets (~2 GB) into the centralized cache (`~/.cache/cvlization/stable-worldmodel/assets`)
2. Reconstruct and load the LeJEPA world model from `lejepa_weights.ckpt`
3. Run a synthetic forward pass to verify the architecture
4. Run a 200-step expert policy rollout in DMControl cheetah and save a video

### What to expect

- **First run**: ~2 GB download (world model checkpoint + expert policy weights)
- **Output**: `stable_worldmodel_outputs/env_0.mp4` — cheetah locomotion video
- **World model load**: prints parameter count (~27M), epoch, keys loaded
- **Verification**: prints predicted embedding shape `(1, 192)` and norm

Run the quality evaluation flags to see the world model in action:

```bash
./predict.sh "python predict.py --skip-rollout --demo-video --tsne --latent-video --skip-download"
```

For a live rollout with world model predictions alongside:

```bash
./predict.sh "python predict.py --skip-rollout --wm-rollout --skip-download"
```

## Output videos and how to interpret them

All outputs land in `stable_worldmodel_outputs/`.

### `env_0.mp4` — expert policy rollout

A plain MuJoCo render of the cheetah running for 200 steps under the SAC expert policy.
The world model is **not** involved. This is the environment baseline — it shows what the task
looks like and confirms the policy and simulator are working.

### `wm_rollout.mp4` — world model prediction during live rollout (2 columns)

```
| actual next frame | WM predicted (NN) |
```

The expert policy runs live in MuJoCo. At every step, the world model sees the last 3 frames
plus the actions taken and predicts the embedding of the *next* frame. That predicted embedding
is matched against the dataset to find the closest real frame (nearest-neighbour retrieval),
which is shown on the right.

**What to look for:** if the right column tracks the left column — similar body pose,
similar leg positions — the world model has learned to predict future states. With the current
undertrained checkpoint (epoch 52/100) the match is loose; a fully-trained model would track
much more closely.

**Why NN retrieval and not direct pixel generation?** JEPA by design predicts in *embedding
space*, not pixel space. There is no decoder. NN retrieval is the standard evaluation method
used in the I-JEPA and V-JEPA papers — it makes the latent prediction visible without a decoder.

### `nn_demo.mp4` — world model prediction on a dataset episode (5 columns)

```
| t-2 | t-1 | t | actual t+1 | WM predicted (NN) |
```

Same idea as `wm_rollout.mp4` but run on a pre-recorded dataset episode instead of a live
rollout. Columns 1–3 are the 3 context frames fed to the world model. Column 4 is what
actually happened next. Column 5 is the world model's prediction via NN retrieval.

**Compared to `wm_rollout.mp4`:** the content is nearly identical (same cheetah, same policy),
but this version explicitly shows the 3-frame context window the model uses.

### `latent_video.mp4` — animated latent space alongside actual frames (2 columns)

```
| actual frame | t-SNE latent space map |
```

Left: actual video frames. Right: the same episode plotted in 2D (t-SNE of all frame
embeddings). The **faded blue trail** is the full episode trajectory. The **bright blue dot**
is the current frame. The **red star** is where the world model predicted the current frame
would be. The **salmon line** connects each prediction to its ground-truth target.

**What to look for:** a well-trained model would have the red star land near the bright blue
dot — short salmon lines. With the current checkpoint the lines are long (predictions are
scattered), confirming the undertrained state.

**This is the closest output to `visualize_trajectories.py`** in the original repo, which
shows the same left-panel + latent-space-right-panel layout.

### `tsne_embeddings.png` — static version of the latent space map

The same data as the right panel of `latent_video.mp4` but as a single static image. Useful
for a quick quality check without playing a video.

### Skip the rollout (world model only)

```bash
./predict.sh "python predict.py --skip-rollout"
```

### Point to already-downloaded assets

```bash
./predict.sh "python predict.py --skip-download --asset-dir /cvl-cache/stable-worldmodel/assets"
```

## HuggingFace Repos

| Repo | Type | Used by this example |
|---|---|---|
| `zzsi/swm-dmc-cheetah` | model | Yes — `lejepa_weights.ckpt` loaded at inference |
| `zzsi/swm-dmc-expert-policies` | model | Yes — SAC expert policy for rollout video |
| `zzsi/swm-dmc-expert` | dataset | Yes — episode frames/actions for `--tsne` and `--demo-video` |
| `zzsi/swm-dmc-mixed-small` | dataset | No — for training only |
| `zzsi/swm-dmc-mixed-large` | dataset | No — for training only |

The `mixed-small` and `mixed-large` datasets contain lower-quality and diverse trajectories intended
for training a world model from scratch (alongside the expert data). They are not needed for
inference and are not downloaded by default.

## How World Model Loading Works

### Why the object checkpoint cannot be used

The official inference path uses `lejepa_epoch_50_object.ckpt`, which was pickled against a private training codebase containing a top-level `module` package (classes: `ARPredictor`, `ConditionalBlock`, `JEPA`, `MLP`, `Attention`, …). This package is absent from every released `stable-worldmodel` tag, so `torch.load(weights_only=False)` raises:

```
No module named 'module'
```

This is a checkpoint/code-version incompatibility, not a download corruption issue. We confirmed this by:
- Inspecting the pickle data inside `lejepa_epoch_50_object.ckpt` — it references 7 classes under the `module.*` namespace
- Checking all released upstream tags (`v0.0.1a0` through `0.0.5`) — none contain a `module.py` file
- Attempting load at runtime — same `ModuleNotFoundError` in all released versions

### Our fix: reconstruct from `lejepa_weights.ckpt`

`lejepa_weights.ckpt` is a plain PyTorch Lightning training checkpoint (`{"epoch": …, "state_dict": {"model.*": tensor, …}}`). It has no pickle class-path dependencies.

We reconstruct the full model architecture in `predict.py` from scratch by matching the exact weight key names and tensor shapes in the checkpoint:

| Component | Architecture | Source |
|---|---|---|
| `encoder` | ViT-tiny: hidden=192, 12 layers, 3 heads, patch_size=14, img=224 | `config.yaml` + weight shapes |
| `action_encoder` | Conv1d(30→10) → Linear(10→768) → SiLU → Linear(768→192) | weight key names |
| `predictor` | 6-layer DiT-style transformer, 16 heads, dim_head=64, mlp_dim=2048 | weight shapes |
| `pred_proj` / `projector` | Linear(192→2048) → BatchNorm → ReLU → Linear(2048→192) | weight shapes |

The predictor uses **adaLN-zero** (adaptive LayerNorm) conditioning: each block has `adaLN_modulation = Sequential(SiLU, Linear(192, 1152))` which outputs 6 factors (shift/scale/gate for attention + MLP) that condition frame predictions on actions.

Loading is done with `model.load_state_dict(model_sd, strict=False)` — `strict=False` is needed because the checkpoint also contains training-only heads (decoder, sigreg) that we don't need at inference.

### Action conditioning format

The action encoder expects a 5-step × 6-dim action window (30-dim total) per frame, passed as `(B, T, 30)`. With `history_size=3` frames, the input is `(B, 3, 30)`.

## Running Other Checks

```bash
# Verify remote HuggingFace repo contents
./predict.sh "python verify_hf_hub.py --strict"

# Download all dataset splits
./predict.sh "python download_assets.py --target-dir /cvl-cache/stable-worldmodel/assets --splits expert mixed-small mixed-large"

# Inspect archive layout
./predict.sh "python inspect_assets.py --asset-dir /cvl-cache/stable-worldmodel/assets"
```

## Build with a different upstream ref

```bash
docker build --build-arg STABLE_WORLDMODEL_REF=v0.0.5 -t cvl-stable-worldmodel-dmc:latest -f Dockerfile .
```

## Attribution

The world model checkpoint (`lejepa_weights.ckpt`), expert policy, and dataset in the HuggingFace repos
[`zzsi/swm-dmc-cheetah`](https://huggingface.co/zzsi/swm-dmc-cheetah),
[`zzsi/swm-dmc-expert-policies`](https://huggingface.co/zzsi/swm-dmc-expert-policies), and
[`zzsi/swm-dmc-expert`](https://huggingface.co/datasets/zzsi/swm-dmc-expert)
are derived from the [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) project.

Please cite the original work:

```bibtex
@misc{maes_lelidec2026swm-1,
      title={stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation},
      author={Lucas Maes and Quentin Le Lidec and Dan Haramati and
              Nassim Massaudi and Damien Scieur and Yann LeCun and
              Randall Balestriero},
      year={2026},
      eprint={2602.08968},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.08968},
}
```
