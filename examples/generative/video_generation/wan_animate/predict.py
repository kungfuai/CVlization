#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from cvlization.paths import resolve_input_path, resolve_output_path


WAN_ROOT = Path("/opt/Wan2.2")
PREPROCESS_SCRIPT = WAN_ROOT / "wan/modules/animate/preprocess/preprocess_data.py"
GENERATE_SCRIPT = WAN_ROOT / "generate.py"


def run_command(args: list[str]) -> None:
    print(" ".join(args))
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if "/opt/Wan2.2" not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{pythonpath}:/opt/Wan2.2" if pythonpath else "/opt/Wan2.2"
    subprocess.run(args, check=True, env=env)


def ensure_checkpoint(ckpt_dir: Path, process_ckpt: Path) -> None:
    if ckpt_dir.exists() and process_ckpt.exists():
        return
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id="Wan-AI/Wan2.2-Animate-14B",
        local_dir=str(ckpt_dir),
        local_dir_use_symlinks=False,
        token=token,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Wan2.2 Animate docker runner.")
    parser.add_argument("--mode", choices=["animate", "replace"], default="animate")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--reference-image", required=True, help="Reference image path.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--ckpt-dir", default="/models/Wan2.2-Animate-14B")
    parser.add_argument(
        "--process-ckpt-dir",
        default="",
        help="Path to process_checkpoint (defaults to <ckpt-dir>/process_checkpoint).",
    )
    parser.add_argument("--resolution-width", type=int, default=1280)
    parser.add_argument("--resolution-height", type=int, default=720)
    parser.add_argument("--refert-num", type=int, default=1)
    parser.add_argument("--use-flux", dest="use_flux", action="store_true")
    parser.add_argument("--no-use-flux", dest="use_flux", action="store_false")
    parser.set_defaults(use_flux=True)
    parser.add_argument("--retarget-flag", dest="retarget_flag", action="store_true")
    parser.add_argument("--no-retarget-flag", dest="retarget_flag", action="store_false")
    parser.set_defaults(retarget_flag=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--w-len", type=int, default=1)
    parser.add_argument("--h-len", type=int, default=1)
    parser.add_argument("--use-relighting-lora", dest="use_relighting_lora", action="store_true")
    parser.add_argument("--no-use-relighting-lora", dest="use_relighting_lora", action="store_false")
    parser.set_defaults(use_relighting_lora=True)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--preprocess-args", nargs="*", default=[])
    parser.add_argument("--generate-args", nargs="*", default=[])

    args = parser.parse_args()

    video_path = Path(resolve_input_path(args.video))
    ref_path = Path(resolve_input_path(args.reference_image))
    output_dir = Path(args.output_dir)  # Output path doesn't need resolution
    output_dir.mkdir(parents=True, exist_ok=True)
    process_dir = output_dir / "process_results"
    process_dir.mkdir(parents=True, exist_ok=True)

    process_ckpt = Path(args.process_ckpt_dir) if args.process_ckpt_dir else Path(args.ckpt_dir) / "process_checkpoint"
    ensure_checkpoint(Path(args.ckpt_dir), process_ckpt)

    if not args.skip_preprocess:
        preprocess_cmd = [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--ckpt_path",
            str(process_ckpt),
            "--video_path",
            str(video_path),
            "--refer_path",
            str(ref_path),
            "--save_path",
            str(process_dir),
            "--resolution_area",
            str(args.resolution_width),
            str(args.resolution_height),
        ]
        if args.mode == "animate":
            if args.retarget_flag:
                preprocess_cmd.append("--retarget_flag")
            if args.use_flux:
                preprocess_cmd.append("--use_flux")
        else:
            preprocess_cmd.extend(
                [
                    "--iterations",
                    str(args.iterations),
                    "--k",
                    str(args.k),
                    "--w_len",
                    str(args.w_len),
                    "--h_len",
                    str(args.h_len),
                    "--replace_flag",
                ]
            )
        preprocess_cmd.extend(args.preprocess_args)
        run_command(preprocess_cmd)

    if not args.skip_generate:
        generate_cmd = [
            sys.executable,
            str(GENERATE_SCRIPT),
            "--task",
            "animate-14B",
            "--ckpt_dir",
            str(args.ckpt_dir),
            "--src_root_path",
            str(process_dir),
            "--refert_num",
            str(args.refert_num),
        ]
        if args.mode == "replace":
            generate_cmd.append("--replace_flag")
            if args.use_relighting_lora:
                generate_cmd.append("--use_relighting_lora")
        generate_cmd.extend(args.generate_args)
        run_command(generate_cmd)

    print(f"Outputs written under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
