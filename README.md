# CVlization: Curated AI Training and Inference Recipes That Just Work

A curated collection of 190+ state-of-the-art open source AI capabilities, packaged in self-contained Docker environments. Find a recipe, test it, copy what you need.

*CVlization stands on the shoulders of giants - we package and test amazing open source projects so you can use them with confidence.*

## Quick Start

```bash
git clone --depth 1 https://github.com/kungfuai/CVlization
cd CVlization

# Install CLI (optional - or just use bash scripts)
pip install .

# Optional: install with remote execution support
pip install .[remote]    # SSH runner
pip install .[aws]       # SageMaker runner
pip install .[deploy]    # Serverless deployment (Cerebrium)

# Browse examples
cvl list                  # compact grouped overview
cvl list -k gpt           # search by keyword
cvl list --format list    # detailed view
# or browse examples/ on GitHub

# Run any example
cvl run nanogpt build
cvl run nanogpt train

# Copy into your project (bundled with cvlization)
cvl export perception/image_classification/torch -o your-project/
```

That's it! Each example is self-contained with its own Dockerfile and dependencies. (We battled CUDA versions and dependency conflicts so you don't have to.)

## Table of Contents

- [Examples](#examples)
- [Running an Example](#running-an-example)
- [Benefits](#benefits-of-using-cvlization)
- [Requirements](#requirements)
- [For Contributors](#for-contributors)
- [License](#licenses)

## Examples

Our `examples/` directory is organized by capability (what the model does) rather than modality (what data it processes).

### Directory Structure

```
examples/
  analytical/          # Prediction & forecasting (time series, tabular ML)
  perception/          # Understand signals (vision, speech, multimodal)
  generative/          # Create content (text, images, video, audio, avatars)
  physical/            # Robotics & embodied AI (vision-language-action models)
  agentic/             # AI agents (RAG, tool use, optimization, workflows)
```

### Catalog of Examples

**Legend:** ✅ = Tested and maintained | 🧪 = Experimental

#### 🔍 Perception (Understanding Signals)

| Capability | Example Directory | Implementations | Status |
|------------|-------------------|-----------------|--------|
| ![Image Classification](./doc/images/plant_classification.png) Image Classification | [`examples/perception/image_classification`](./examples/perception/image_classification) | torch, cifar10-speedrun | ✅ |
| ![Object Detection](./doc/images/object_detection.jpg) Object Detection | [`examples/perception/object_detection`](./examples/perception/object_detection) | mmdet, torchvision, rt-detr, yolov13 | ✅ |
| ![Segmentation](./doc/images/semantic_segmentation.png) Segmentation | [`examples/perception/segmentation`](./examples/perception/segmentation) | instance (mmdet, torchvision), semantic (mmseg, torchvision), panoptic (detectron2, mmdet, torchvision), sam, sam_lora_finetuning, sam3, sam3_finetuning | ✅ |
| ![Pose Estimation](./doc/images/pose_estimation.jpeg) Pose Estimation | [`examples/perception/pose_estimation`](./examples/perception/pose_estimation) | dwpose, mmpose | ✅ |
| ![Object Tracking](./doc/images/player_tracking.gif) Tracking | [`examples/perception/tracking`](./examples/perception/tracking) | global_tracking_transformer, soccer_visual_tracking | ✅ |
| ![Line Detection](./doc/images/line_detection.png) Line Detection | [`examples/perception/line_detection`](./examples/perception/line_detection) | torch | ✅ |
| ![Document AI](./doc/images/layoutlm.png) Document AI | [`examples/perception/doc_ai`](./examples/perception/doc_ai) | OCR (chandra_ocr, deepseek_ocr, docling, doctr, dots_ocr, nanonets_ocr, olmocr_2, paddleocr_vl, surya), VLMs (donut_doc_classification, donut_doc_parse, granite_docling, granite_docling_finetune), Layout (doclayout_yolo), Parsing (dolphin_v2, churro_3b, extract0, nvidia_nemotron_parse), Leaderboard (leaderboard) | ✅ |
| ![Vision-Language](./doc/images/layoutlm.png) Vision-Language Models | [`examples/perception/vision_language`](./examples/perception/vision_language) | florence_2 (+ finetune), gemma3_vision (+ grpo, sft), internvl3, joycaption_llava, kosmos2_grounded_ocr, lighton_ocr, llama3_vision, llava_next_video, minicpm_v_2_6, molmoe_1b, moondream2 (+ finetune), moondream3, owl_vit, paligemma2 (detection, segmentation), phi_3_5_vision_instruct, phi_4_multimodal_instruct, pixtral_12b, qwen3_vl | ✅ |
| ![3D: rendering and reconstruction](./doc/images/nerf.gif) 3D Reconstruction | [`examples/perception/3d_reconstruction`](./examples/perception/3d_reconstruction) | dust3r, mast3r, monst3r, hunyuanworld_mirror, map_anything, nerf_tf (experimental) | ✅ |

#### ✨ Generative (Creating Content)

| Capability | Example Directory | Implementations | Status |
|------------|-------------------|-----------------|--------|
| ![LLMs](./doc/images/llm.png) LLMs (text generation) | [`examples/generative/llm`](./examples/generative/llm) | Pretraining (nanogpt, modded_nanogpt, nanomamba), Full pipeline (nanochat: pretrain, sft, rl), Fine-tuning (peft_mistral7b_sft, trl_sft, miles_grpo, unsloth: gpt_oss_grpo, gpt_oss_sft, llama_3b_sft, qwen_7b_sft, gemma3_4b_sft), Inference (mixtral8x7b, sglang, vllm, dllm, nanbeige4_3b_thinking, nomos_1, rnj_1_instruct), Interpretability (gemma_scope_2_270m_it) | ✅ |
| ![Image Generation](./doc/images/controlnet.png) Image Generation | [`examples/generative/image_generation`](./examples/generative/image_generation) | cfm, ddpm, diffuser_unconditional, dit, dreambooth, edm2, flux, next_scene_qwen, qwen_image_layered, rae, repa, stable_diffusion, uva_energy (experimental), vqgan | ✅ |
| ![Video Generation](./doc/images/sora.gif) Video Generation | [`examples/generative/video_generation`](./examples/generative/video_generation) | animate_diff, animate_diff_cog, animate_x, cogvideox, deforum, flashvsr, framepack, hunyuan_video_1_5, kandinsky_5, krea_realtime_scope (experimental), longcat_video, ltx2, mimic_motion, minisora, phantom (experimental), propainter, real_video, reward_forcing, skyreals, svd_cog, svd_comfy, turbodiffusion, vace, vace_comfy (experimental), video2x, video_enhancement, video_in_between, wan2gp, wan2gp_wan, wan_animate, wan_comfy, worldcanvas | ✅ |
| Text-to-Speech (TTS) | [`examples/generative/audio`](./examples/generative/audio) | cosyvoice3, vibevoice_realtime, voxcpm | ✅ |
| Avatar & Talking Head | [`examples/generative/video_generation/avatar`](./examples/generative/video_generation/avatar) | anytalker, egstalker, fastavatar, flashportrait, hunyuanvideo_avatar, imtalker, lite_avatar, live_avatar, livetalk, longcat_video_avatar, personalive, wan_s2v | ✅ |

#### 📊 Analytical (Prediction & Forecasting)

| Capability | Example Directory | Implementations | Status |
|------------|-------------------|-----------------|--------|
| **Time Series Forecasting** | [`examples/analytical/time_series`](./examples/analytical/time_series) | Foundation models (chronos_zero_shot, moirai_zero_shot, uni2ts_finetune (experimental)), Supervised (patchtst_supervised), Statistical baselines (statsforecast_baselines), Hierarchical (hierarchical_reconciliation), Anomaly detection (anomaly_transformer, merlion_anomaly_dashboard) | ✅ |
| **Tabular ML - AutoML** | [`examples/analytical/tabular/automl`](./examples/analytical/tabular/automl) | autogluon_structured, pycaret_structured | ✅ |
| **Tabular ML - Causal Inference** | [`examples/analytical/tabular/causal`](./examples/analytical/tabular/causal) | causalml_campaign_optimization, dowhy_berkeley_bias, dowhy_policy_uplift, econml_heterogeneous_effects | ✅ |
| **Tabular ML - Uncertainty Quantification** | [`examples/analytical/tabular/uncertainty`](./examples/analytical/tabular/uncertainty) | Conformal (conformal_lightgbm, mapie_conformal), Quantile (catboost_quantile, quantile_lightgbm), Bayesian (pymc_bayesian_regression) | ✅ |
| **Tabular ML - Business Use Cases** | [`examples/analytical/tabular`](./examples/analytical/tabular) | Customer analytics (gbt_telco_churn), Marketing (gbt_upsell_propensity), Risk scoring (gbt_credit_default), Regression (gbt_housing_prices), Recommendation (ranking_lightgbm), Survival (pycox_retention), Anomaly detection (pyod_fraud_detection), Feature engineering (autofe_structured) | ✅ |

#### 🦾 Physical (Robotics & Embodied AI) - Experimental

| Capability | Example Directory | Implementations | Status |
|------------|-------------------|-----------------|--------|
| **Vision-Language-Action Models** | [`examples/physical`](./examples/physical) | openvla_single_step, openvla_simplerenv | 🧪 |

#### 🤖 Agentic (AI Agents & Workflows) - Experimental

| Capability | Example Directory | Implementations | Status |
|------------|-------------------|-----------------|--------|
| **RAG & Knowledge** | [`examples/agentic/rag`](./examples/agentic/rag) | langgraph_helpdesk, clara | 🧪 |
| **LlamaIndex Agents** | [`examples/agentic/llamaindex`](./examples/agentic/llamaindex) | graph_rag_cognee, jsonalyze_structured_qa, react_finance_query_agent | 🧪 |
| **Prompt Optimization** | [`examples/agentic/optimization`](./examples/agentic/optimization) | dspy_gepa_promptops, mcts_prompt_agent | 🧪 |
| **Tool Use & Coding** | [`examples/agentic`](./examples/agentic) | Code (autogen_pair_programmer), Data analysis (smolagents_data_analyst), Data preparation (physio_signal_prep), Local AI (llamacpp_assistant) | 🧪 |
| **Formal Reasoning** | [`examples/agentic/formal`](./examples/agentic/formal) | nanoproof (Lean theorem proving) | 🧪 |

Note: These examples are regularly updated and tested to ensure compatibility with the latest dependencies. We recommend starting with the nanogpt example.

**Browse on GitHub:** [perception/](./examples/perception/) • [generative/](./examples/generative/) • [analytical/](./examples/analytical/) • [physical/](./examples/physical/) • [agentic/](./examples/agentic/)

**Browse via CLI:** `cvl list` for a compact overview, or `cvl list -k <keyword>` to search

## Running an Example

You can run examples using the `cvl` CLI or directly with bash scripts.

**Option 1: Using cvl CLI (recommended)**

```bash
cvl run nanogpt build
cvl run nanogpt train
```

**Option 2: Using bash scripts directly**

```bash
cd examples/generative/llm/nanogpt
bash build.sh
bash train.sh
```

**More examples:**

```bash
# RL post-training with GRPO (cutting-edge reasoning model training)
cvl run miles_grpo train

# Generate video from text prompt (Tencent HunyuanVideo)
cvl run hunyuan-video-1-5 predict --prompt "A cat playing piano"

# Document AI extraction (IBM Granite-Docling)
cvl run granite-docling predict -i input_pdf=@document.pdf

# RAG agent with LangGraph
cvl run agentic-rag-langgraph-helpdesk predict --question "How do I list examples?"
```

For detailed instructions and available options, see the README.md in each example directory.

**License Note:** Each example may reference projects with different licenses. Check the license file in each example directory.

## Remote Execution (Experimental)

Run examples on cloud infrastructure instead of locally:

```bash
# AWS SageMaker (managed training)
cvl run nanogpt train --runner sagemaker --spot --output-path s3://bucket/outputs

# SkyPilot (any cloud: AWS, GCP, Azure, Lambda Labs)
cvl run nanogpt train --runner skypilot --gpu A100:1 --cloud aws

# SSH (existing GPU server)
cvl run nanogpt train --runner ssh user@gpu-host
```

Manage remote jobs:
```bash
cvl jobs list --runner sagemaker
cvl jobs logs <job-id>
cvl jobs kill <job-id>
```

See [cvl/runners/README.md](./cvl/runners/README.md) for setup instructions.

## Serverless Deployment (Experimental)

Deploy inference endpoints to serverless GPU platforms:

```bash
# Deploy to Cerebrium
cvl deploy ltx2 --gpu L40

# Manage deployed services
cvl services list
cvl services status ltx2
cvl services logs ltx2
cvl services delete ltx2
```

See [cvl/deployers/README.md](./cvl/deployers/README.md) for setup and supported platforms.

## Benefits of Using CVlization

**Centralized Caching:** All examples use `~/.cache/cvlization/` for models and datasets, avoiding re-downloads across examples:
- Automatic caching for HuggingFace Hub, PyTorch, and custom downloads
- Managed by build scripts - no manual setup required
- Shared across all examples to save disk space and bandwidth

**Self-Contained Docker Environments:** Each example is isolated with pinned dependencies:
- CUDA and dependency conflicts already resolved - saves hours or days of setup
- Source code is mounted at runtime (not baked into images) for easy iteration
- Dependency versions are pinned where possible for reproducibility

**Production-Ready Patterns:** Copy what works into your projects:
- Consistent build/train/predict script structure across all examples
- Battle-tested configurations for 190+ AI capabilities
- Examples regularly updated and tested for compatibility

## Requirements

- Docker ([Install Docker](https://docs.docker.com/get-docker/))
- NVIDIA GPU (most examples require 16GB+ VRAM; A10 or better recommended)
- nvidia-docker for GPU access
  ```bash
  # Ubuntu
  sudo apt-get install -y nvidia-container-toolkit
  ```

## Use CVlization on Colab

No installation needed - run examples directly in Google Colab:
[Colab notebook: CIFAR-10 classification](https://colab.research.google.com/drive/1FkZcZnJC_z-PuFSYM91kU1-d63-LecMJ?usp=sharing)

## For Contributors

CVlization includes Claude Code skills for AI-assisted development and automated verification:

- **`verify-training-pipeline`** - Validates training examples are properly structured, build successfully, train without errors, and log appropriate metrics
- **`verify-inference-example`** - Validates inference examples build correctly and run inference successfully

These skills enable Claude to automatically verify examples end-to-end, helping maintain code quality across the repository.

**Contributing Guidelines:**
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for standardization patterns and best practices
- All examples should follow the build/train/predict script pattern
- Use verification metadata in `example.yaml` to track testing status

## Project Structure

- `examples/`: Dockerized AI examples (perception, generative, analytical, physical, agentic)
- `cvl/`: CLI tool source code
- `cvlization/`: Optional reusable library components
- `doc/`: Project documentation, including `doc/runners/` for cloud setup guides
- `tests/`: Unit and integration tests

## CVlization Library

The `cvlization/` directory provides optional reusable components:

**Available:**
- Training pipeline abstractions (image classification, object detection, LLMs, diffusion)
- Dataset builders with caching (PyTorch, TensorFlow, HuggingFace)
- Model factories (Torchvision, MMDetection, MMSegmentation)
- Utilities (metrics, logging, download helpers)

**Installation:** `pip install -e .`

**Note:** Examples are self-contained and don't require the library. For production use, copying example code directly is often simpler than depending on the library package.

## Documentation

Detailed documentation can be found in the `doc/` directory:

- [Computer vision model training workflow and quality checks](./doc/archived/Computer%20vision%20model%20training%20workflow%20and%20quality%20checks.pdf)
- [Multi-task multi-input models: a common pattern](./doc/archived/Multi-task%20multi-input%20models_%20a%20common%20pattern.pdf)
- [Reusable model components](./doc/archived/reusable_model_components.md)

## Licenses

**CVlization Library & CLI:** MIT License
- The `cvlization` package and `cvl` CLI tool are released under the MIT License
- Safe for commercial use

**Examples Directory:** Mixed Licenses
- Examples may reference projects with various licenses (copyleft, non-commercial, etc.)
- Examples are NOT included when you `pip install cvlization`
- Always check the license file in each example directory before using in production

**Note:** Each example packages different open-source projects with their own licenses. Review licenses carefully for your use case.
