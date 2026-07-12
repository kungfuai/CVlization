#!/usr/bin/env python3
"""
Lance unified multimodal inference — text-to-image, image understanding, and
image editing from a single 3B-parameter model.

Wraps the upstream bytedance/Lance inference pipeline for CVlization.
Model weights are pulled from bytedance-research/Lance on HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LANCE_REPO = Path("/opt/lance")
SUPPORTED_TASKS = {
    "t2i": "Text-to-image generation",
    "x2t_image": "Image understanding (VQA)",
    "image_edit": "Image editing with text instruction",
}

# Default generation parameters
DEFAULT_RESOLUTION = 768
DEFAULT_NUM_STEPS = 30
DEFAULT_CFG_SCALE = 4.0
DEFAULT_TIMESTEP_SHIFT = 3.5


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download Lance model from HuggingFace and return the snapshot path."""
    from huggingface_hub import snapshot_download

    print(f"[lance] Ensuring model weights from {model_id} ...", flush=True)
    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        # Skip video model and assets to save bandwidth/disk
        ignore_patterns=["Lance_3B_Video/*", "assets/*", "*.md"],
    )
    print(f"[lance] Model snapshot at: {snapshot_path}", flush=True)
    return Path(snapshot_path)


def setup_downloads_symlink(snapshot_path: Path) -> None:
    """Create downloads/ symlink in the Lance repo so default paths resolve."""
    downloads_dir = LANCE_REPO / "downloads"
    if downloads_dir.is_symlink() or downloads_dir.exists():
        if downloads_dir.is_symlink():
            downloads_dir.unlink()
        else:
            shutil.rmtree(downloads_dir)

    # The HF snapshot has Lance_3B/, Qwen2.5-VL-ViT/, Wan2.2_VAE.pth at root
    downloads_dir.symlink_to(snapshot_path)
    print(f"[lance] Linked {downloads_dir} -> {snapshot_path}", flush=True)


def write_t2i_examples(prompt: str, work_dir: Path) -> Path:
    """Write a text-to-image example JSON for the Lance pipeline."""
    examples = {"000000.png": prompt}
    json_path = work_dir / "t2i_input.json"
    json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    return json_path


def write_x2t_image_examples(
    image_path: str, question: str, work_dir: Path
) -> Path:
    """Write an image-understanding example JSON for the Lance pipeline."""
    examples = {
        "0001": {
            "interleave_array": [
                image_path,
                [
                    "Look at the image carefully and answer the question.",
                    question,
                    "",
                ],
            ],
            "element_dtype_array": ["image", "text"],
            "istarget_in_interleave": [0, 1],
        }
    }
    json_path = work_dir / "x2t_image_input.json"
    json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    return json_path


def write_image_edit_examples(
    image_path: str, instruction: str, work_dir: Path
) -> Path:
    """Write an image-editing example JSON for the Lance pipeline."""
    examples = {
        "0001": {
            "interleave_array": [
                instruction,
                image_path,
                "",  # output placeholder
            ],
            "element_dtype_array": ["text", "image", "image"],
            "istarget_in_interleave": [0, 0, 1],
        }
    }
    json_path = work_dir / "image_edit_input.json"
    json_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    return json_path


def run_lance_inference(
    task: str,
    example_json: Path,
    output_dir: Path,
    model_path: str,
    resolution: int,
    num_steps: int,
    cfg_scale: float,
    timestep_shift: float,
) -> None:
    """Invoke the upstream Lance inference_lance.py via accelerate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use accelerate launch for proper TrainingArguments compatibility
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--mixed_precision", "bf16",
        str(LANCE_REPO / "inference_lance.py"),
        "--model_path", model_path,
        "--vit_type", "qwen_2_5_vl_original",
        "--llm_qk_norm", "true",
        "--llm_qk_norm_und", "true",
        "--llm_qk_norm_gen", "true",
        "--tie_word_embeddings", "false",
        "--validation_num_timesteps", str(num_steps),
        "--validation_timestep_shift", str(timestep_shift),
        "--copy_init_moe", "true",
        "--max_num_frames", "121",
        "--max_latent_size", "64",
        "--latent_patch_size", "1", "1", "1",
        "--visual_und", "true",
        "--visual_gen", "true",
        "--vae_model_type", "wan",
        "--apply_qwen_2_5_vl_pos_emb", "true",
        "--apply_chat_template", "false",
        "--cfg_type", "0",
        "--validation_data_seed", "42",
        "--video_height", str(resolution),
        "--video_width", str(resolution),
        "--num_frames", "1",
        "--task", task,
        "--save_path_gen", str(output_dir),
        "--resolution", "image_768res",
        "--text_template", "true",
        "--cfg_text_scale", str(cfg_scale),
        "--use_KVcache", "true",
        "--val_dataset_config_file", str(example_json),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(LANCE_REPO) + ":" + env.get("PYTHONPATH", "")

    print(f"[lance] Running {task} inference ...", flush=True)
    result = subprocess.run(cmd, env=env, cwd=str(LANCE_REPO))
    if result.returncode != 0:
        print(f"[lance] Inference exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)


def collect_results(output_dir: Path, task: str, final_dir: Path) -> None:
    """Copy generated outputs to the final artifacts directory."""
    final_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in sorted(output_dir.iterdir()):
        if f.suffix in (".png", ".jpg", ".mp4", ".json"):
            dest = final_dir / f.name
            shutil.copy2(f, dest)
            results.append(str(dest))

    if task == "x2t_image":
        result_json = output_dir / "result.json"
        if result_json.exists():
            data = json.loads(result_json.read_text())
            print("\n[lance] Understanding results:", flush=True)
            for entry in data:
                q = entry.get("question", "")
                a = entry.get("answer", "")
                print(f"  Q: {q}", flush=True)
                print(f"  A: {a}", flush=True)

    if results:
        print(f"\n[lance] Outputs saved to {final_dir}:", flush=True)
        for r in results:
            print(f"  {r}", flush=True)
    else:
        print(f"\n[lance] No outputs found in {output_dir}", flush=True)

    # Write metrics.json
    metrics = {
        "task": task,
        "num_outputs": len(results),
        "output_files": results,
    }
    metrics_path = final_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[lance] Metrics written to {metrics_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance unified multimodal inference (CVlization wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks:
  t2i          Generate an image from a text prompt
  x2t_image    Answer a question about an image (VQA)
  image_edit   Edit an image given a text instruction

Examples:
  # Text-to-image
  python predict.py --task t2i --prompt "A cat wearing a top hat"

  # Image understanding
  python predict.py --task x2t_image --input-image photo.jpg --prompt "What is in this image?"

  # Image editing
  python predict.py --task image_edit --input-image photo.jpg --edit-instruction "Add sunglasses"
""",
    )
    parser.add_argument(
        "--task",
        choices=list(SUPPORTED_TASKS),
        default="t2i",
        help="Inference task (default: t2i)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean with vibrant orange and purple colors",
        help="Text prompt for t2i, or question for x2t_image",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Input image path (required for x2t_image and image_edit)",
    )
    parser.add_argument(
        "--edit-instruction",
        type=str,
        default=None,
        help="Edit instruction (for image_edit task)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Directory for final outputs (default: ./artifacts)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="bytedance-research/Lance",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Image resolution in pixels (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of denoising steps (default: {DEFAULT_NUM_STEPS})",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=DEFAULT_CFG_SCALE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_CFG_SCALE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (default: auto)",
    )
    args = parser.parse_args()

    # Validate task-specific args
    if args.task in ("x2t_image",) and not args.input_image:
        parser.error(f"--input-image is required for task '{args.task}'")
    if args.task == "image_edit" and not args.input_image:
        parser.error("--input-image is required for image_edit")
    if args.task == "image_edit" and not args.edit_instruction:
        parser.error("--edit-instruction is required for image_edit")

    # Resolve input image via cvlization paths if available
    input_image = args.input_image
    if input_image:
        try:
            from cvlization.paths import resolve_input_path
            input_image = resolve_input_path(input_image)
        except ImportError:
            input_image = os.path.abspath(input_image)
        if not os.path.isfile(input_image):
            parser.error(f"Input image not found: {input_image}")

    # Download model
    snapshot_path = download_model(args.model_id)
    setup_downloads_symlink(snapshot_path)

    model_path = str(snapshot_path / "Lance_3B")

    # Create task-specific example JSON
    work_dir = Path(tempfile.mkdtemp(prefix="lance_"))
    gen_output_dir = work_dir / "results"

    if args.task == "t2i":
        example_json = write_t2i_examples(args.prompt, work_dir)
    elif args.task == "x2t_image":
        example_json = write_x2t_image_examples(
            input_image, args.prompt, work_dir
        )
    elif args.task == "image_edit":
        example_json = write_image_edit_examples(
            input_image, args.edit_instruction, work_dir
        )
    else:
        parser.error(f"Unsupported task: {args.task}")

    print(f"[lance] Task: {args.task} ({SUPPORTED_TASKS[args.task]})", flush=True)
    print(f"[lance] Model: {model_path}", flush=True)
    print(f"[lance] Resolution: {args.resolution}x{args.resolution}", flush=True)
    print(f"[lance] Steps: {args.num_steps}, CFG: {args.cfg_scale}", flush=True)

    # Run inference
    run_lance_inference(
        task=args.task,
        example_json=example_json,
        output_dir=gen_output_dir,
        model_path=model_path,
        resolution=args.resolution,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        timestep_shift=DEFAULT_TIMESTEP_SHIFT,
    )

    # Collect results — resolve output dir via cvlization paths for host persistence
    final_dir = Path(args.output_dir)
    try:
        from cvlization.paths import resolve_output_path
        resolved = resolve_output_path(
            args.output_dir.rstrip("/") + "/", default_filename="output"
        )
        final_dir = Path(resolved).parent
    except ImportError:
        final_dir = Path(os.path.abspath(args.output_dir))

    collect_results(gen_output_dir, args.task, final_dir)

    # Cleanup temp dir
    shutil.rmtree(work_dir, ignore_errors=True)
    print("[lance] Done.", flush=True)


if __name__ == "__main__":
    main()
