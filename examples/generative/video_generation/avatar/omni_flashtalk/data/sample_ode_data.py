"""Generate Stage-1 KD targets: run SoulX-FlashTalk over every manifest item.

For each (portrait, audio, prompt) the SoulX-FlashTalk-14B teacher produces a
talking-avatar video. That MP4 is the Stage-1 knowledge-distillation target —
kept in pixel space so it is VAE-agnostic (the Stage-1 trainer re-encodes each
video with the student's own VAE).

Runs the already-built `soulx_flashtalk` docker image, one container per item,
sharded across GPUs with a worker pool. Resume-safe: skips items whose output
MP4 already exists.

Usage (on acasia):
    python3 sample_ode_data.py --data-dir ~/zz/omni_flashtalk_data \
        --gpus 0,1,2,3,4 --hf-cache ~/.cache/huggingface
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time


def run_item(item, data_dir, gpu, hf_cache, log_dir):
    """Run SoulX on one item, pinned to one GPU."""
    item_id = item["id"]
    out_mp4 = os.path.join(data_dir, "soulx_targets", f"{item_id}.mp4")
    if os.path.exists(out_mp4):
        return item_id, "skip", 0.0
    t0 = time.time()
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
        "--mount", f"type=bind,src={hf_cache},dst=/root/.cache/huggingface",
        "--mount", f"type=bind,src={data_dir},dst=/data",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "soulx_flashtalk", "python3", "predict.py",
        "--audio", f"/data/{item['audio_path']}",
        "--image", f"/data/{item['image_path']}",
        "--output", f"/data/soulx_targets/{item_id}.mp4",
        "--prompt", item["full_prompt"],
    ]
    log_path = os.path.join(log_dir, f"{item_id}.log")
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    ok = rc == 0 and os.path.exists(out_mp4)
    return item_id, ("ok" if ok else "FAIL"), dt


def worker(gpu, work_q, results, data_dir, hf_cache, log_dir):
    while True:
        try:
            item = work_q.get_nowait()
        except queue.Empty:
            return
        item_id, status, dt = run_item(item, data_dir, gpu, hf_cache, log_dir)
        results.append((item_id, status, dt))
        print(f"[gpu{gpu}] {item_id}: {status} ({dt:.0f}s)  "
              f"[{len(results)} done]", flush=True)
        work_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4")
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    hf_cache = os.path.abspath(os.path.expanduser(args.hf_cache))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    os.makedirs(os.path.join(data_dir, "soulx_targets"), exist_ok=True)
    log_dir = os.path.join(data_dir, "soulx_logs")
    os.makedirs(log_dir, exist_ok=True)

    manifest = os.path.join(data_dir, "manifest_assets.jsonl")
    items = [json.loads(l) for l in open(manifest) if l.strip()]
    items = [it for it in items if it.get("image_path") and it.get("audio_path")]
    print(f"{len(items)} items, {len(gpus)} GPUs ({gpus})")

    work_q = queue.Queue()
    for it in items:
        work_q.put(it)
    results = []
    threads = [
        threading.Thread(target=worker,
                         args=(g, work_q, results, data_dir, hf_cache, log_dir))
        for g in gpus
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for _, s, _ in results if s == "ok")
    skip = sum(1 for _, s, _ in results if s == "skip")
    fail = [i for i, s, _ in results if s == "FAIL"]
    print(f"\nDone in {(time.time()-t0)/60:.1f} min: "
          f"{ok} ok, {skip} skipped, {len(fail)} failed")
    if fail:
        print(f"  failed ids: {fail}")


if __name__ == "__main__":
    main()
