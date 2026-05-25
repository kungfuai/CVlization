"""Streaming inference v7: combines v5's torchrun pipeline parallelism
with v6's classifier-free guidance. Each of N=4 ranks owns one denoising
step; per step, the rank runs TWO model forward passes (positive +
negative conditioning) and combines via CFG before passing to the next
rank.

Per-tick cost: ~2 × forward = ~788 ms (vs v5's ~388 ms without CFG).
Quality target: matches v6 (sharp output, latent std ~ 0.7).
Throughput target: ~0.84 s of video per 788 ms tick = ~1.07x real-time.

Run with:
    torchrun --nproc-per-node 4 --standalone streaming_v7_pp_cfg.py \\
        --num-chunks 8 --cfg-scale 3.0
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--item-id", default="00000")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--wan-vae", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--num-chunks", type=int, default=8)
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-on-last-rank", action="store_true",
                    help="if set, last rank does VAE decode after step 4 "
                         "(adds ~891 ms to rank 3's tick; throughput bottleneck). "
                         "If not set, only save raw clean latents (defer decode).")
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v7")
    args = ap.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)
    is_last = (rank == world_size - 1)

    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        OmniAudioCausalWanModel, load_omni_into_causal_adapter,
        load_wan_vae, decode_latent,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK,
        DENOISING_STEPS, SIGMAS)
    from PIL import Image

    S = world_size
    assert S == len(DENOISING_STEPS), \
        f"need {len(DENOISING_STEPS)} ranks (one per denoising step), got {S}"

    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[v7] {S} ranks, {args.num_chunks} chunks, cfg={args.cfg_scale}, "
              f"item={args.item_id}", flush=True)

    # ---- Load model on this rank ------------------------------------------
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    lat = torch.load(os.path.join(args.data_dir, "latents", f"{args.item_id}.pt"),
                     map_location="cpu", weights_only=False)
    audio_full = lat["audio_emb"].float()
    text = lat["text_context"].float()
    ref = lat["ref_latent"].float()
    H, W = ref.shape[2], ref.shape[3]
    F_CHUNK = NUM_FRAME_PER_BLOCK
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3
    seq_len = F_CHUNK * H * W // 4

    ref_dev = ref[:, 0].unsqueeze(0).to(device, torch.bfloat16)
    text_dev = text.unsqueeze(0).to(device, torch.bfloat16)
    text_neg = torch.zeros_like(text_dev)
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    # Pre-generated noise (broadcast to all ranks)
    noise_shape = (args.num_chunks, S, 1, 16, F_CHUNK, H, W)
    noise_all = torch.empty(noise_shape, device=device, dtype=torch.bfloat16)
    if is_master:
        g = torch.Generator(device=device).manual_seed(args.seed)
        for c in range(args.num_chunks):
            for s in range(S):
                noise_all[c, s] = torch.randn(
                    (1, 16, F_CHUNK, H, W),
                    generator=g, device=device, dtype=torch.bfloat16)
    dist.broadcast(noise_all, src=0)

    sigma_cur = SIGMAS[rank]
    sigma_next = SIGMAS[rank + 1] if rank < S - 1 else None
    t_val = DENOISING_STEPS[rank]

    def fwd(noisy, audio_dev, txt):
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [txt[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)

    def run_step(noisy, audio_dev, audio_neg, chunk_idx):
        """CFG step: positive + negative forward, combine, x0, optional re-noise."""
        pred_pos = fwd(noisy, audio_dev, text_dev)
        pred_neg = fwd(noisy, audio_neg, text_neg)
        flow_pred = pred_neg + args.cfg_scale * (pred_pos - pred_neg)
        x0 = noisy - sigma_cur * flow_pred
        x0[:, :, :args.motion_latent_frames] = motion
        if not is_last:
            new_noise = noise_all[chunk_idx, rank + 1]
            new_noisy = (1 - sigma_next) * x0 + sigma_next * new_noise
            new_noisy[:, :, :args.motion_latent_frames] = motion
            return new_noisy
        return x0  # clean

    # ---- Pipeline ---------------------------------------------------------
    total_ticks = args.num_chunks + S
    completed_chunks = []
    if is_master:
        tick_times = []
        t_start = time.time()

    for tick in range(total_ticks):
        torch.cuda.synchronize(device)
        t_tick = time.time()

        chunk_idx = tick - rank
        has_work = (0 <= chunk_idx < args.num_chunks)

        if has_work:
            if rank == 0:
                noisy = noise_all[chunk_idx, 0].clone()
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                noisy = torch.empty(
                    (1, 16, F_CHUNK, H, W), device=device, dtype=torch.bfloat16)
                dist.recv(noisy, src=rank - 1)

            a0 = chunk_idx * F_CHUNK * 4
            audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
            if audio_chunk.shape[0] < T_AUDIO_CHUNK:
                audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
            audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)
            audio_neg = torch.zeros_like(audio_dev)

            result = run_step(noisy, audio_dev, audio_neg, chunk_idx)

            if not is_last:
                dist.send(result.contiguous(), dst=rank + 1)
            else:
                completed_chunks.append((chunk_idx, result.cpu()))

        torch.cuda.synchronize(device)
        if is_master:
            tick_times.append(time.time() - t_tick)
            print(f"  tick {tick:2d}  dt={tick_times[-1]*1000:.0f}ms  "
                  f"r0_chunk={chunk_idx if has_work else 'idle'}",
                  flush=True)

    dist.barrier()

    if is_master:
        total = time.time() - t_start
        # Skip warmup ticks AND idle drain ticks
        nontrivial = [t for t in tick_times if t > 0.05]
        steady = sum(nontrivial[S:]) / max(1, len(nontrivial) - S) if len(nontrivial) > S else (sum(nontrivial) / max(1, len(nontrivial)))
        print(f"\n[v7] DONE: {args.num_chunks} chunks in {total:.2f}s", flush=True)
        print(f"  steady tick: {steady*1000:.0f} ms", flush=True)
        chunk_video_sec = (4 * F_CHUNK - 3) / 25  # 21 pixel frames / 25 fps
        # Effective throughput at steady state
        eff_chunks_per_sec = 1.0 / steady
        rt = eff_chunks_per_sec * chunk_video_sec
        print(f"  chunk emits {4*F_CHUNK - 3} pixel frames = {chunk_video_sec:.2f} s of video", flush=True)
        print(f"  throughput at steady tick: {rt:.2f}x real-time", flush=True)
        # Latency = time to first finished chunk = S ticks
        first_chunk_lat = sum(tick_times[:S]) if len(tick_times) >= S else total
        print(f"  latency to first chunk: {first_chunk_lat*1000:.0f} ms", flush=True)

    if is_last and args.decode_on_last_rank:
        vae = load_wan_vae(args.wan_vae, device)
        print(f"[v7 rank {rank}] decoding {len(completed_chunks)} chunks...", flush=True)
        # Decode every chunk to its full pixel-frame sequence, save all frames
        # as numbered PNGs, then ffmpeg-mux them into a single mp4.
        frames_dir = os.path.join(args.out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        global_idx = 0
        completed_chunks.sort(key=lambda x: x[0])  # ensure chunk order
        for chunk_idx, clean_cpu in completed_chunks:
            clean = clean_cpu.to(device, dtype=torch.bfloat16)
            pix = decode_latent(vae, clean.squeeze(0), device)  # [3, F_pix, H, W]
            F_pix = pix.shape[1]
            for f_idx in range(F_pix):
                img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                Image.fromarray(img).save(
                    os.path.join(frames_dir, f"frame{global_idx:05d}.png"))
                global_idx += 1
        print(f"[v7 rank {rank}] saved {global_idx} frames to {frames_dir}", flush=True)

        # ffmpeg-mux the frames into mp4
        import subprocess
        mp4_path = os.path.join(args.out_dir, "video.mp4")
        cmd = [
            "ffmpeg", "-y", "-framerate", "25",
            "-i", os.path.join(frames_dir, "frame%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            mp4_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[v7 rank {rank}] wrote {mp4_path} ({global_idx} frames @ 25 fps)",
                  flush=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[v7 rank {rank}] ffmpeg failed: {e}; frames saved at {frames_dir}",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
