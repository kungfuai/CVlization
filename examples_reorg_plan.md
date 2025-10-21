# Examples Reorganization Plan

## New Structure

```
examples/
  analytical/          # Prediction on structured/unstructured data
  perception/          # Understand signals (CV/ASR/OCR/VL)
  generative/          # Create content (LLMs, images, audio, video)
  agentic/             # Planning, tools, RAG, workflows (future)
```

## Migration Mapping

### PERCEPTION (31 examples)

#### perception/ocr_and_layout/ (4 examples)
- `doc_ai/docling-serve` → `perception/ocr_and_layout/docling_serve`
- `doc_ai/dots-ocr` → `perception/ocr_and_layout/dots_ocr`
- `doc_ai/nanonets-ocr` → `perception/ocr_and_layout/nanonets_ocr`
- `doc_ai/surya` → `perception/ocr_and_layout/surya`

#### perception/vision_language/ (7 examples)
- `doc_ai/granite-docling` → `perception/vision_language/granite_docling`
- `doc_ai/granite-docling-finetune` → `perception/vision_language/granite_docling_finetune`
- `doc_ai/huggingface/donut/doc_classification` → `perception/vision_language/donut_doc_classification`
- `doc_ai/huggingface/donut/doc_parse` → `perception/vision_language/donut_doc_parse`
- `doc_ai/moondream2` → `perception/vision_language/moondream2`
- `doc_ai/moondream2_finetune` → `perception/vision_language/moondream2_finetune`
- `doc_ai/moondream3` → `perception/vision_language/moondream3`

#### perception/object_detection/ (2 examples)
- `object_detection/mmdet` → `perception/object_detection/mmdet`
- `object_detection/torchvision` → `perception/object_detection/torchvision`

#### perception/segmentation/ (6 examples)
- `instance_segmentation/mmdet` → `perception/segmentation/instance_mmdet`
- `instance_segmentation/sam` → `perception/segmentation/sam`
- `instance_segmentation/torchvision` → `perception/segmentation/instance_torchvision`
- `panoptic_segmentation/mmdet` → `perception/segmentation/panoptic_mmdet`
- `panoptic_segmentation/torchvision` → `perception/segmentation/panoptic_torchvision`
- `semantic_segmentation/mmseg` → `perception/segmentation/semantic_mmseg`
- `semantic_segmentation/torchvision` → `perception/segmentation/semantic_torchvision`

#### perception/pose_estimation/ (2 examples)
- `pose_estimation/dwpose` → `perception/pose_estimation/dwpose`
- `pose_estimation/mmpose` → `perception/pose_estimation/mmpose`

#### perception/tracking/ (2 examples)
- `object_tracking/global_tracking_transformer` → `perception/tracking/global_tracking_transformer`
- `sports/soccer_game_visual_tracking` → `perception/tracking/soccer_visual_tracking`

#### perception/image_classification/ (1 example)
- `image_classification/torch` → `perception/image_classification/torch`

#### perception/line_detection/ (1 example)
- `line_detection/torch` → `perception/line_detection/torch`

#### perception/3d_reconstruction/ (1 example)
- `nerf/tf` → `perception/3d_reconstruction/nerf_tf`

### GENERATIVE (31 examples)

#### generative/text_generation/ (10 examples)
- `text_gen/mistral7b` → `generative/text_generation/mistral7b`
- `text_gen/mixtral8x7b` → `generative/text_generation/mixtral8x7b`
- `text_gen/modded-nanogpt` → `generative/text_generation/modded_nanogpt`
- `text_gen/modded-nanogpt-original` → `generative/text_generation/modded_nanogpt_original`
- `text_gen/nanochat` → `generative/text_generation/nanochat`
- `text_gen/nanogpt` → `generative/text_generation/nanogpt`
- `text_gen/nanomamba` → `generative/text_generation/nanomamba`
- `text_gen/trl/sft` → `generative/text_generation/trl_sft`
- `text_gen/unsloth/gpt_oss_grpo` → `generative/text_generation/unsloth_gpt_oss_grpo`
- `text_gen/unsloth/gpt_oss_sft` → `generative/text_generation/unsloth_gpt_oss_sft`
- `text_gen/unsloth/llama_3b_sft` → `generative/text_generation/unsloth_llama_3b_sft`
- `text_gen/unsloth/qwen_7b_sft` → `generative/text_generation/unsloth_qwen_7b_sft`

#### generative/image_generation/ (10 examples)
- `image_gen/cfm` → `generative/image_generation/cfm`
- `image_gen/ddpm` → `generative/image_generation/ddpm`
- `image_gen/diffuser_unconditional` → `generative/image_generation/diffuser_unconditional`
- `image_gen/dit` → `generative/image_generation/dit`
- `image_gen/dreambooth` → `generative/image_generation/dreambooth`
- `image_gen/edm2` → `generative/image_generation/edm2`
- `image_gen/flux` → `generative/image_generation/flux`
- `image_gen/mdt` → `generative/image_generation/mdt`
- `image_gen/pixart` → `generative/image_generation/pixart`
- `image_gen/vqgan` → `generative/image_generation/vqgan`

#### generative/video_generation/ (11 examples)
- `video_gen/animate_diff_cog` → `generative/video_generation/animate_diff_cog`
- `video_gen/animate_x` → `generative/video_generation/animate_x`
- `video_gen/deforum` → `generative/video_generation/deforum`
- `video_gen/framepack` → `generative/video_generation/framepack`
- `video_gen/kandinsky` → `generative/video_generation/kandinsky`
- `video_gen/mimic_motion` → `generative/video_generation/mimic_motion`
- `video_gen/minisora` → `generative/video_generation/minisora`
- `video_gen/phantom` → `generative/video_generation/phantom`
- `video_gen/skyreals` → `generative/video_generation/skyreals`
- `video_gen/svd_cog` → `generative/video_generation/svd_cog`
- `video_gen/vace` → `generative/video_generation/vace`
- `video_gen/vace_comfy` → `generative/video_generation/vace_comfy`
- `video_gen/video_in_between` → `generative/video_generation/video_in_between`
- `video_gen/wan2gp` → `generative/video_generation/wan2gp`
- `video_gen/wan_comfy` → `generative/video_generation/wan_comfy`

### ANALYTICAL (0 examples currently)
- Placeholder for future NLP classification/extraction tasks

### AGENTIC (0 examples currently)
- Placeholder for future RAG/tool-use/planning examples

## Migration Script

The migration will be done in phases to avoid breaking things:

### Phase 1: Create new structure
```bash
mkdir -p examples/analytical
mkdir -p examples/perception/{ocr_and_layout,vision_language,object_detection,segmentation,pose_estimation,tracking,image_classification,line_detection,3d_reconstruction}
mkdir -p examples/generative/{text_generation,image_generation,video_generation}
mkdir -p examples/agentic
```

### Phase 2: Move examples (preserving git history)
All moves will use `git mv` to preserve history.

### Phase 3: Clean up empty old directories
After all moves are complete, remove empty old category directories.

### Phase 4: Update references
- Update any scripts that reference old paths
- Update documentation
- Update CI/CD if it exists

## Notes

- **Naming convention**: Use underscores in folder names (e.g., `donut_doc_classification` not `donut-doc-classification`)
- **Flatten nested examples**: No more `huggingface/donut/` nesting - becomes `donut_doc_classification`
- **Keep build failures**: Examples like `diffuser_unconditional`, `mdt`, `kandinsky`, `minisora`, `wan2gp` will move but remain broken until fixed
- **Cog examples preserved**: `animate_diff_cog`, `deforum`, `svd_cog` keep their cog.yaml setup

## Total Count
- **Perception**: 31 examples
- **Generative**: 31 examples
- **Analytical**: 0 examples (future)
- **Agentic**: 0 examples (future)
- **Total migrating**: 62 examples
