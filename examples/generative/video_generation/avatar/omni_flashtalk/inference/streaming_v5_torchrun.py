"""Streaming inference, Phase 5: multiprocess pipeline parallelism with
deterministic noise. Fixes the two execution issues in v4:

1. **Python GIL bottleneck** (v4: 2256ms tick): each GPU runs in its own
   process via torchrun, so model.forward() calls don't fight for the GIL.
   Communication between adjacent ranks uses NCCL P2P send/recv on
   GPU-resident tensors.

2. **Noise determinism** (v4: identical re-noise tensors across steps): all
   noise tensors for all chunks are pre-generated on rank 0 with one master
   seeded generator, then broadcast to every rank. Each rank picks the
   slice it needs by (chunk_idx, step_idx).

   This also makes v5 numerically comparable to v2 (sequential reference)
   when both use the same master noise schedule.

Schedule (S=4 ranks, one per denoising step):
  tick t  rank 0 (step 1)  rank 1 (step 2)  rank 2 (step 3)  rank 3 (step 4)
    0     chunk 0          -                -                -
    1     chunk 1          chunk 0          -                -
    2     chunk 2          chunk 1          chunk 0          -
    3     chunk 3          chunk 2          chunk 1          chunk 0
    4     chunk 4          chunk 3          chunk 2          chunk 1
    ...                                                       (emit each tick)

After 3-tick warmup, every tick emits one finished chunk.

Run with:
    torchrun --nproc-per-node 4 --standalone streaming_v5_torchrun.py \\
        --num-chunks 8
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v5")
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
        print(f"[v5] {S} ranks, {args.num_chunks} chunks, item={args.item_id}", flush=True)

    # ---- Load model on this rank ------------------------------------------
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, args.wan_base, args.omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # ---- Load item (every rank loads — small file) ------------------------
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
    motion = ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(
        device, torch.bfloat16)

    # ---- Pre-generate ALL noise on rank 0, broadcast to all ranks ---------
    # Per chunk we need S+1 noise tensors:
    #   noise[c, 0]   = initial noise for step 1 (sigma=1.0)
    #   noise[c, k]   = re-noise tensor injected between step k and step k+1
    #                   (sigma transition from SIGMAS[k-1] to SIGMAS[k])
    # noise[c, S] is unused (last step doesn't re-noise).
    noise_shape = (args.num_chunks, S, 1, 16, F_CHUNK, H, W)
    noise_all = torch.empty(noise_shape, device=device, dtype=torch.bfloat16)
    if is_master:
        g = torch.Generator(device=device).manual_seed(args.seed)
        # Fill in chunk/step order so the sequence matches a single-GPU loop
        for c in range(args.num_chunks):
            for s in range(S):
                noise_all[c, s] = torch.randn(
                    (1, 16, F_CHUNK, H, W),
                    generator=g, device=device, dtype=torch.bfloat16)
    dist.broadcast(noise_all, src=0)
    if is_master:
        print(f"[v5] noise pre-generated + broadcast: shape={tuple(noise_all.shape)}",
              flush=True)

    sigma_cur = SIGMAS[rank]
    sigma_next = SIGMAS[rank + 1] if rank < S - 1 else None
    t_val = DENOISING_STEPS[rank]

    def run_step(noisy, audio_dev, chunk_idx):
        """Run rank's assigned denoising step. Returns either next-noisy
        (if not last rank) or clean (if last rank)."""
        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=device, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_dev.unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [text_dev[0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            flow_pred = model(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)
        x0 = noisy - sigma_cur * flow_pred
        x0[:, :, :args.motion_latent_frames] = motion

        if not is_last:
            # Re-noise with the PRE-GENERATED noise for (chunk_idx, rank+1)
            new_noise = noise_all[chunk_idx, rank + 1]
            new_noisy = (1 - sigma_next) * x0 + sigma_next * new_noise
            new_noisy[:, :, :args.motion_latent_frames] = motion
            return new_noisy
        return x0  # clean

    # ---- Pipeline driver ---------------------------------------------------
    # State on this rank: incoming noisy tensor for current tick, or None.
    # Communication: every tick, rank > 0 receives from rank - 1, rank < S-1
    # sends to rank + 1. Audio is local (every rank has audio_full).
    total_ticks = args.num_chunks + S  # ensure last chunk gets all S steps + 1 drain
    completed_chunks = []  # only populated on last rank
    if is_master:
        tick_times = []
        t_start = time.time()

    # Each rank also tracks which chunk_idx is currently in its slot.
    # rank 0 enqueues a new chunk per tick from tick 0 to tick num_chunks-1.
    # rank r processes chunk c at tick (c + r).
    for tick in range(total_ticks):
        torch.cuda.synchronize(device)
        t_tick = time.time()

        # Determine which chunk_idx (if any) should be processed at this rank, this tick
        chunk_idx = tick - rank
        has_work = (0 <= chunk_idx < args.num_chunks)

        if has_work:
            if rank == 0:
                # Build fresh noisy from pre-generated noise + motion anchor
                noisy = noise_all[chunk_idx, 0].clone()
                noisy[:, :, :args.motion_latent_frames] = motion
            else:
                # Receive from previous rank
                noisy = torch.empty(
                    (1, 16, F_CHUNK, H, W), device=device, dtype=torch.bfloat16)
                dist.recv(noisy, src=rank - 1)

            # Per-chunk audio slice
            a0 = chunk_idx * F_CHUNK * 4
            audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
            if audio_chunk.shape[0] < T_AUDIO_CHUNK:
                audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
            audio_dev = audio_chunk.unsqueeze(0).to(device, torch.bfloat16)

            result = run_step(noisy, audio_dev, chunk_idx)

            if not is_last:
                dist.send(result.contiguous(), dst=rank + 1)
            else:
                completed_chunks.append((chunk_idx, result.cpu()))

        torch.cuda.synchronize(device)
        if is_master:
            tick_times.append(time.time() - t_tick)
            print(f"  tick {tick:2d}  dt={tick_times[-1]*1000:.0f}ms  "
                  f"rank0_chunk={chunk_idx if has_work else 'idle'}",
                  flush=True)

    dist.barrier()

    if is_master:
        total = time.time() - t_start
        avg = sum(tick_times) / len(tick_times)
        steady = sum(tick_times[S:]) / max(1, len(tick_times) - S)
        print(f"\n[v5] DONE: {args.num_chunks} chunks in {total:.2f}s", flush=True)
        print(f"  avg tick:    {avg*1000:.0f}ms", flush=True)
        print(f"  steady tick: {steady*1000:.0f}ms  (= 1 chunk per tick at steady state)", flush=True)
        print(f"  throughput:  {args.num_chunks/total:.2f} chunks/s "
              f"= {args.num_chunks * F_CHUNK * 4 / 25 / total:.2f}x real-time "
              f"(target > 1.0)", flush=True)

    # ---- Decode + save on last rank ---------------------------------------
    if is_last:
        vae = load_wan_vae(args.wan_vae, device)
        print(f"[v5 rank {rank}] decoding {len(completed_chunks)} chunks...", flush=True)
        for chunk_idx, clean_cpu in completed_chunks:
            clean = clean_cpu.to(device, dtype=torch.bfloat16)
            pix = decode_latent(vae, clean.squeeze(0), device)
            F_pix = pix.shape[1]
            for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
                img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                Image.fromarray(img).save(
                    os.path.join(args.out_dir, f"chunk{chunk_idx:02d}_{fl}.png"))
        print(f"[v5 rank {rank}] saved {len(completed_chunks)*3} PNGs to {args.out_dir}",
              flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
