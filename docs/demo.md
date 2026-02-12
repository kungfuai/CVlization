# CVlization Demo Guide

This guide is a practical demo plan using `cvl run` pipelines in this repo.

## How To Run Any Pipeline

Use this pattern:

```bash
cvl run <pipeline-name> build
cvl run <pipeline-name> <preset>
```

Common presets are `train`, `predict`, `evaluate`, `serve`, and `smoke-test`.

## Recommended Demo Portfolio (Balanced Across 5 Areas)

These are good defaults for a live demo with broad coverage and manageable risk.

| Area | Pipeline | Why it demos well | Typical command flow |
|---|---|---|---|
| analytical | `churn-gbt` | Full train + metrics + predict, CPU-friendly | `build -> train -> predict` |
| analytical | `pyod-fraud` | Strong anomaly-detection story, class imbalance handling | `build -> train -> predict` |
| analytical | `merlion-anomaly-dashboard` | Interactive monitoring/dashboard UX | `build -> smoke-test -> serve` |
| generative | `diffuser-unconditional` | Visual output, familiar diffusion workflow | `build -> train` |
| generative/llm | `nanogpt` | From-scratch GPT training, small footprint | `build -> train` |
| perception | `moondream2` | OCR + caption + VQA in one model | `build -> predict` |
| agentic | `agentic-smolagents-data-analyst` | Tool-using agent over tabular data | `build -> predict -> evaluate` |
| physical | `openvla-simplerenv` | End-to-end robot policy demo in simulation | `build -> run` |

## Perception Coverage (Expanded)

If you want deeper computer-vision coverage, use this set:

1. `image-classification-torch` (classification): `build -> train`
2. `object-detection-torchvision` (detection): `build -> train`
3. `semantic-segmentation-torchvision` (segmentation): `build -> train`
4. `soccer-visual-tracking` (tracking): `build -> predict`
5. `moondream2` (vision-language OCR/VQA): `build -> predict`

Suggested command examples:

```bash
cvl run image-classification-torch build
cvl run image-classification-torch train

cvl run object-detection-torchvision build
cvl run object-detection-torchvision train

cvl run semantic-segmentation-torchvision build
cvl run semantic-segmentation-torchvision train

cvl run soccer-visual-tracking build
cvl run soccer-visual-tracking predict

cvl run moondream2 build
cvl run moondream2 predict
```

## 3D Coverage (Expanded)

For 3D reconstruction demos, pick by available GPU memory:

1. `mast3r` (recommended first): 8GB+ VRAM, `build -> predict`
2. `dust3r`: 16GB+ VRAM, `build -> predict`
3. `map_anything`: 16GB+ VRAM, `build -> predict`
4. `hunyuanworld_mirror`: 20GB+ VRAM, `build -> predict`
5. `monst3r`: very heavy (80GB VRAM), `build -> predict`
6. `nerf-tf`: legacy educational training example, `build -> train`

Suggested command examples:

```bash
cvl run mast3r build
cvl run mast3r predict

cvl run dust3r build
cvl run dust3r predict

cvl run map_anything build
cvl run map_anything predict
```

## What About `nanochat` and `nanogpt`?

Both are good demos, but they fit different situations.

### `nanogpt` (recommended default LLM demo)

- Best for: showing from-scratch language model training without extreme hardware.
- Resource profile: ~1 GPU / 8GB VRAM.
- Flow:

```bash
cvl run nanogpt build
cvl run nanogpt train
```

### `nanochat` (advanced/high-end demo)

- Best for: showing full ChatGPT-style pipeline (pretrain, SFT, RL stages).
- Resource profile in pipeline metadata is very high (8 GPUs, 80GB VRAM each for full-scale flow).
- Use when you have serious compute and want to showcase the full stack.
- Flow:

```bash
cvl run nanochat build
cvl run nanochat train
# or
cvl run nanochat train-full
```

## Suggested Demo Tracks

## Track A: 45-Minute Reliable Demo (low risk)

1. `churn-gbt`: `build -> train -> predict`
2. `pyod-fraud`: `build -> train -> predict`
3. `moondream2`: `build -> predict`
4. `agentic-smolagents-data-analyst`: `build -> predict`

## Track B: Vision + 3D Heavy Demo

1. `image-classification-torch`: `build -> train`
2. `object-detection-torchvision`: `build -> train`
3. `semantic-segmentation-torchvision`: `build -> train`
4. `mast3r` or `dust3r`: `build -> predict`

## Track C: LLM-Focused Demo

1. `nanogpt`: `build -> train`
2. `nanochat`: `build -> train` (if hardware allows)
3. Optional serving/agent add-on: `agentic-rag-langgraph-helpdesk`

## Reality Checks

1. Not every pipeline has the same maturity. Some are inference-only, some are full lifecycle.
2. GPU-heavy demos (`nanochat`, some video/3D pipelines) can fail on underprovisioned machines.
3. For live sessions, always keep one CPU-friendly fallback from analytical or agentic categories.

## Example Notes (What Each Demo Answers)

Use these bullets while presenting each pipeline. They are aligned to setup, data prep, training/eval/tuning, and production-readiness questions.

### `churn-gbt` (analytical/tabular/customer_analytics)

- Task/domain: binary classification for churn prediction (tabular).
- Skill level: beginner-to-intermediate (easy commands, moderate metric interpretation).
- Data prep: downloads Telco CSV, drops ID column, coerces bad `TotalCharges`, median-imputes blanks, stratified split.
- Sample efficiency: works with ~7k rows; no special few-shot method, just a compact tabular dataset.
- Data quality handling: explicit coercion/imputation for known dirty numeric field.
- Training feedback: LightGBM logs, early stopping, train/val/test metrics.
- Evaluation metrics: accuracy, ROC-AUC, macro F1, classification report.
- Tuning levers: `num_boost_round`, `early_stopping_rounds`, train/validation split ratio.
- Fixed parts: GBT/LightGBM family, target mapping, baseline preprocessing pattern.
- Export/deploy: model artifacts saved, `predict` preset consumes saved model + input CSV.
- Pre-prod validation: file existence checks + holdout metrics artifact.
- Monitoring: no built-in live monitoring loop in this pipeline.

### `pyod-fraud` (analytical/tabular/anomaly_detection)

- Task/domain: anomaly detection / fraud detection on highly imbalanced tabular data.
- Skill level: intermediate (anomaly metrics like PR-AUC matter).
- Data prep: downloads `creditcard.csv`, separates normal/fraud, down-samples majority class.
- Sample efficiency: controllable via `MAX_NORMAL_SAMPLES` to reduce compute while preserving rare fraud samples.
- Data quality handling: standardization (`StandardScaler`) and contamination estimation from training data.
- Training feedback: contamination estimate and train completion logs.
- Evaluation metrics: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, classification report.
- Tuning levers: `MAX_NORMAL_SAMPLES`, `TEST_SIZE`, Isolation Forest hyperparameters (e.g. estimators).
- Fixed parts: Isolation Forest detector class + scaler-based preprocessing.
- Export/deploy: detector/scaler serialized; `predict` preset scores new records.
- Pre-prod validation: artifact existence checks and batch prediction output verification.
- Monitoring: no always-on monitoring service; good offline scoring artifacts for audit.

### `merlion-anomaly-dashboard` (analytical/time_series/anomaly_dashboard)

- Task/domain: time-series anomaly exploration and dashboard-based analysis.
- Skill level: beginner for UI demo, intermediate for detector interpretation.
- Data prep: mount CSV files with timestamp/value columns or use NAB-like datasets.
- Sample efficiency: depends on dataset and detector chosen in dashboard; not fixed by one trainer script here.
- Data quality handling: not deeply specified in this example; depends on loaded series.
- Training feedback: primarily UI-driven; this example emphasizes serving and smoke testing.
- Evaluation metrics: available via Merlion workflows, but this repo example focuses on dashboard operation.
- Tuning levers: detector/forecaster choices and dashboard configuration.
- Fixed parts: Dash + gunicorn serving pattern.
- Export/deploy: `serve` launches production-style web app process.
- Pre-prod validation: explicit `smoke-test` preset checks app endpoints.
- Monitoring: this is the strongest built-in monitoring/exploration style demo in analytical examples.

### `diffuser-unconditional` (generative/image_generation)

- Task/domain: unconditional image generation with diffusion.
- Skill level: intermediate (GPU + diffusion training basics).
- Data prep: dataset choice via pipeline config; standard image dataset workflow.
- Sample efficiency: no explicit sample-efficient claim; depends on dataset size/steps.
- Data quality handling: mostly inherited from dataset/transforms in training code.
- Training feedback: training loss progression from diffusers training loop.
- Evaluation metrics: primarily qualitative image quality unless extra eval tooling is added.
- Tuning levers: training steps, batch size, LR, scheduler/model config.
- Fixed parts: diffusion framework and unconditional-generation paradigm.
- Export/deploy: trained checkpoints/images as artifacts.
- Pre-prod validation: mainly training-completion and sample-generation sanity checks.
- Monitoring: no built-in online monitoring service.

### `nanogpt` (generative/llm)

- Task/domain: from-scratch GPT text generation training.
- Skill level: intermediate.
- Data prep: small text corpus workflow (e.g., Shakespeare in example context).
- Sample efficiency: good for demos because dataset and model can be tiny.
- Data quality handling: minimal in this simple benchmark-style setup.
- Training feedback: iterative train loss logs; checkpoint progression.
- Evaluation metrics: usually train/val loss and generated sample quality.
- Tuning levers: model size/depth, context length, batch size, learning rate, iterations.
- Fixed parts: Karpathy nanoGPT architecture/training style.
- Export/deploy: checkpoints and generation scripts.
- Pre-prod validation: basic loss trend and generation sanity checks.
- Monitoring: none built in.

### `nanochat` (generative/llm)

- Task/domain: full chat-model pipeline (base/mid/SFT/RL stages).
- Skill level: advanced.
- Data prep: stage-specific data pipeline (pretrain + instruction/chat + RL signals).
- Sample efficiency: not a small-data pipeline by default; designed for larger training runs.
- Data quality handling: depends on stage dataset curation; not unified in one simple script.
- Training feedback: stage-by-stage logs/metrics; pipeline completion by stage.
- Evaluation metrics: stage-dependent; may include benchmark/eval scripts.
- Tuning levers: model depth/size, stage arguments, training durations.
- Fixed parts: multi-stage training workflow philosophy.
- Export/deploy: chat/inference scripts and model checkpoints.
- Pre-prod validation: stage completion and eval scripts.
- Monitoring: no integrated production monitoring service in this example.

### `moondream2` (perception/vision_language)

- Task/domain: OCR, image captioning, visual question answering.
- Skill level: beginner-to-intermediate.
- Data prep: for predict demos, direct image input; for fine-tune variant, JSONL with image/question/answer.
- Sample efficiency: model is compact for a VLM; inference demo requires no training data.
- Data quality handling: mostly input-image quality and prompt quality.
- Training feedback: not applicable to inference-only demo; see `moondream2_finetune` for training loops.
- Evaluation metrics: inference outputs are qualitative unless you run benchmark/eval scripts.
- Tuning levers: prompt format, inference params, model variant.
- Fixed parts: underlying moondream model family.
- Export/deploy: direct containerized inference pipeline.
- Pre-prod validation: predictable `predict` flow with explicit input/output paths.
- Monitoring: no built-in monitoring service.

### `image-classification-torch` (perception/image_classification)

- Task/domain: supervised image classification.
- Skill level: intermediate.
- Data prep: code path uses `TorchvisionDatasetBuilder(dataset_classname="CIFAR10")`.
- Dataset used by default: CIFAR10.
- Dataset size: 60,000 images total (50,000 train / 10,000 test, standard CIFAR10 split).
- Repo caveat: current builder `validation_dataset()` also constructs with `train=True`, so train/val are drawn from the training split in this implementation.
- Sample efficiency: typically moderate; can downscale dataset/model for faster demos.
- Data quality handling: transform/augmentation + dataset normalization conventions.
- Training feedback: epoch loss/accuracy and checkpoint logs.
- Evaluation metrics: top-1 accuracy (and often validation loss).
- Tuning levers: backbone, epochs, LR, batch size, augmentations.
- Fixed parts: Torchvision training stack.
- Export/deploy: model checkpoints as artifacts.
- Pre-prod validation: validation-set performance trend.
- Monitoring: no online monitoring provided.

### `object-detection-torchvision` (perception/object_detection)

- Task/domain: object detection (FCOS / Faster R-CNN / RetinaNet family).
- Skill level: intermediate-to-advanced.
- Data prep: detection annotations + transforms; default code path uses `KittiTinyDatasetBuilder`.
- Dataset used by default: KITTI Tiny.
- Dataset size (default split files): 75 images total, with 50 train and 25 val.
- Alternate option in code (commented): Penn-Fudan Pedestrian with 30 train / 20 val in this builder.
- Sample efficiency: small benchmark subsets can demo quickly; full quality needs more data.
- Data quality handling: heavily dependent on label quality and box consistency.
- Training feedback: detector training losses (classification/regression components).
- Evaluation metrics: detection metrics (mAP family) where evaluation scripts are provided.
- Tuning levers: detector backbone/model, LR schedule, epochs, image size.
- Fixed parts: Torchvision detection framework.
- Export/deploy: trained detector weights/checkpoints.
- Pre-prod validation: validation metrics + visual spot checks.
- Monitoring: not integrated.

### `semantic-segmentation-torchvision` (perception/segmentation)

- Task/domain: semantic segmentation.
- Skill level: intermediate-to-advanced.
- Data prep: paired image/mask data with label mapping.
- Dataset used by current `train.py`: Stanford Background (`iccv09Data`), not VOC/COCO.
- Dataset size: 715 images total; builder generates an 80/20 split (about 572 train / 143 val).
- Metadata caveat: `example.yaml` lists VOC/COCO, but training code currently points to Stanford Background.
- Sample efficiency: quick demo possible on smaller subsets; high IoU needs more data.
- Data quality handling: mask quality is critical; bad masks degrade training quickly.
- Training feedback: segmentation train/val loss logs.
- Evaluation metrics: IoU / mIoU and pixel accuracy (depending on scripts).
- Tuning levers: model head/backbone, crop/resize strategy, LR, epochs.
- Fixed parts: Torchvision segmentation setup.
- Export/deploy: trained checkpoint artifacts.
- Pre-prod validation: mIoU trend and qualitative mask review.
- Monitoring: not integrated.

### `panoptic-segmentation-torchvision` (perception/segmentation)

- Task/domain: panoptic-style segmentation from a Torchvision stack.
- Skill level: advanced.
- Dataset used by current `train.py`: Penn-Fudan Pedestrian.
- Dataset size in this builder: 30 train / 20 val.
- Implementation note: uses Mask R-CNN + lightweight semantic head and Detectron2-style panoptic merge logic.
- Data quality handling: depends on instance mask quality (semantic targets are derived from masks).
- Training feedback: detector losses + semantic loss + validation mAP + panoptic segment count.
- Evaluation metrics: detection mAP is built-in; panoptic sanity is currently count/visualization-oriented rather than full PQ benchmark.
- Tuning levers: confidence threshold, overlap threshold, semantic head weight, batch size/steps.
- Fixed parts: Torchvision detector backbone family and merge strategy.
- Export/deploy: training-oriented; no dedicated serving script in this example.
- Monitoring: not integrated.

### `panoptic-segmentation-detectron2` (perception/segmentation)

- Task/domain: Detectron2 PanopticFPN inference/visualization.
- Skill level: intermediate-to-advanced.
- Dataset/model source: COCO-pretrained PanopticFPN from Detectron2 model zoo.
- Data prep: single image input (sample image auto-downloaded when input is omitted).
- Training feedback: not applicable (predict-focused example).
- Evaluation metrics: qualitative visualization output; add COCO panoptic eval for PQ metrics.
- Tuning levers: score threshold and model zoo config/weights.
- Fixed parts: Detectron2 panoptic architecture and post-processing stack.
- Export/deploy: produces panoptic visualization image artifact.
- Monitoring: not integrated.

### `soccer-visual-tracking` (perception/tracking)

- Task/domain: sports object tracking and bird's-eye-view analytics.
- Skill level: intermediate.
- Data prep: video input plus pretrained model weights.
- Sample efficiency: inference-focused demo, no training loop required.
- Data quality handling: robust enough for demo clips; real matches require curated camera conditions.
- Training feedback: not applicable (predict-first pipeline).
- Evaluation metrics: mostly visual/trajectory quality unless custom scoring is added.
- Tuning levers: detection/tracking thresholds and video/model choices.
- Fixed parts: pipeline architecture and pretrained assets.
- Export/deploy: output videos and tracking artifacts.
- Pre-prod validation: smoke-style run on known clip.
- Monitoring: no persistent monitoring service.

### `mast3r` / `dust3r` / `map_anything` (perception/3d_reconstruction)

- Task/domain: feed-forward multi-view 3D reconstruction.
- Skill level: intermediate-to-advanced (3D outputs and camera geometry interpretation).
- Data prep: multi-view image sets of same scene with overlap.
- Sample efficiency: demo works with a handful of images; quality improves with better view coverage.
- Data quality handling: sensitive to blur, low overlap, and dynamic scene motion.
- Training feedback: inference-oriented; no local training epoch loop in these demos.
- Evaluation metrics: mostly qualitative geometry checks (GLB/point cloud/depth consistency) unless you add external benchmarks.
- Tuning levers: reconstruction iteration count, confidence thresholds, image resolution/model variant.
- Fixed parts: pretrained model weights and inference pipeline.
- Export/deploy: GLB/point cloud/depth/confidence outputs are deployment-ready artifacts for downstream tools.
- Pre-prod validation: inspect generated assets on known test scenes.
- Monitoring: no online monitoring built in.

### `openvla-simplerenv` (physical/vla)

- Task/domain: robot-manipulation policy evaluation in simulation.
- Skill level: advanced.
- Data prep: task/environment setup rather than classic tabular/image labeling.
- Sample efficiency: demo is inference/evaluation-oriented; not focused on training from scratch.
- Data quality handling: simulation/task config quality matters more than dataset cleaning.
- Training feedback: not primarily a trainer example; focuses on running policy in env.
- Evaluation metrics: task success / rollout behavior (pipeline-specific).
- Tuning levers: task selection, runtime configs, policy/model choice.
- Fixed parts: OpenVLA + SimplerEnv + web streaming stack.
- Export/deploy: containerized web demo (`run`) suitable for reproducible showcases.
- Pre-prod validation: smoke tests available in pipeline presets.
- Monitoring: interactive web visualization, but not a full production monitoring platform.
