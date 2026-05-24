"""Streaming inference, Phase 4: pipeline parallelism across denoising steps.

Each of N=4 GPUs owns one of the 4 denoising steps permanently. At every
tick, each GPU runs its step on the chunk that's currently at its slot;
output flows to the next GPU's slot for the next tick. After a 3-tick warmup,
one finished chunk pops out per tick.

This is the LiveAvatar-style throughput trick: throughput = 1 chunk per
t_step rather than 1 chunk per S*t_step.

Important caveat: this version uses INDEPENDENT chunks (each anchored to
the same ref image, no cross-chunk attention). Cross-chunk temporal
continuity needs KV cache, which we punted on (v3 issue with LoRA). For
short streams or chunked-talking-head use cases this is fine; for long-form
continuous video it would show seams at chunk boundaries.

Verification target: chunk N's output should match v2 chunk 0's quality
(coherent talking face), AND wall-clock throughput should be ~4x v2 once
the pipeline fills up.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


def load_per_gpu_model(gpu_id, wan_base, omni_lora):
    """Load a model on the given GPU index. Returns the model in eval mode."""
    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        OmniAudioCausalWanModel, load_omni_into_causal_adapter,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK)
    device = f"cuda:{gpu_id}"
    model = OmniAudioCausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_LEN,
        in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=TEXT_DIM,
        out_dim=16, num_heads=12, num_layers=30,
        local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model.num_frame_per_block = NUM_FRAME_PER_BLOCK
    model = load_omni_into_causal_adapter(model, wan_base, omni_lora)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/whadmin/zz/omni_flashtalk_data")
    ap.add_argument("--item-id", default="00000")
    ap.add_argument("--wan-base", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    ap.add_argument("--omni-lora", default="/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt")
    ap.add_argument("--wan-vae", default="/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="comma-separated GPU ids to use (one per denoising step)")
    ap.add_argument("--num-chunks", type=int, default=6,
                    help="total chunks to generate")
    ap.add_argument("--motion-latent-frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/whadmin/zz/omni_flashtalk_data/streaming_v4")
    args = ap.parse_args()

    sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
    sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
    from train_stage1_ode import (
        load_wan_vae, decode_latent,
        TEXT_LEN, TEXT_DIM, NUM_FRAME_PER_BLOCK,
        DENOISING_STEPS, SIGMAS)
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    S = len(gpu_ids)
    assert S == len(DENOISING_STEPS), \
        f"need exactly {len(DENOISING_STEPS)} GPUs (one per denoising step), got {S}"
    torch.manual_seed(args.seed)

    # ---- Load model on each GPU --------------------------------------------
    print(f"loading student model on {S} GPUs: {gpu_ids}...")
    models, devices = [], []
    t_load = time.time()
    for g in gpu_ids:
        m, d = load_per_gpu_model(g, args.wan_base, args.omni_lora)
        models.append(m)
        devices.append(d)
    print(f"  loaded all {S} replicas in {time.time()-t_load:.1f}s")

    # VAE only on the LAST GPU (where final outputs land)
    vae = load_wan_vae(args.wan_vae, devices[-1])

    # ---- Load item ---------------------------------------------------------
    lat = torch.load(os.path.join(args.data_dir, "latents", f"{args.item_id}.pt"),
                     map_location="cpu", weights_only=False)
    audio_full = lat["audio_emb"].float()
    text = lat["text_context"].float()
    ref = lat["ref_latent"].float()
    H, W = ref.shape[2], ref.shape[3]
    F_CHUNK = NUM_FRAME_PER_BLOCK
    T_AUDIO_CHUNK = 4 * F_CHUNK - 3
    seq_len = F_CHUNK * H * W // 4

    # Per-GPU constant tensors (ref, text, motion anchor — copied to each device)
    ref_per_gpu, text_per_gpu, motion_per_gpu = [], [], []
    for d in devices:
        ref_per_gpu.append(ref[:, 0].unsqueeze(0).to(d, torch.bfloat16))
        text_per_gpu.append(text.unsqueeze(0).to(d, torch.bfloat16))
        motion_per_gpu.append(
            ref[:, 0:1].repeat(1, args.motion_latent_frames, 1, 1).unsqueeze(0).to(d, torch.bfloat16))

    def run_step(g_idx, noisy, audio_dev):
        """Run one denoising step on GPU g_idx. Returns (next_noisy, clean_or_None)."""
        d = devices[g_idx]
        m = models[g_idx]
        sigma = SIGMAS[g_idx]
        t_val = DENOISING_STEPS[g_idx]

        mask = torch.zeros(1, 1, F_CHUNK, H, W, device=d, dtype=torch.bfloat16)
        mask[:, :, args.motion_latent_frames:] = 1.0
        ref_tiled = ref_per_gpu[g_idx].unsqueeze(2).expand(-1, -1, F_CHUNK, -1, -1)
        dit_in = [torch.cat([noisy[0], ref_tiled[0], mask[0]], dim=0)]
        ctx = [text_per_gpu[g_idx][0]]
        t_frames = torch.full((1, F_CHUNK), t_val, device=d, dtype=torch.bfloat16)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            flow_pred = m(dit_in, t_frames, ctx, seq_len, audio_emb=audio_dev)
        x0 = noisy - sigma * flow_pred
        x0[:, :, :args.motion_latent_frames] = motion_per_gpu[g_idx]

        if g_idx < S - 1:
            next_sigma = SIGMAS[g_idx + 1]
            new_noise = torch.randn_like(noisy)
            new_noisy = (1 - next_sigma) * x0 + next_sigma * new_noise
            new_noisy[:, :, :args.motion_latent_frames] = motion_per_gpu[g_idx]
            return new_noisy, None
        else:
            return None, x0

    # ---- Pipelined schedule -----------------------------------------------
    # slots[g] = chunk currently being processed by GPU g (None = idle)
    # Each entry is (noisy_latent_on_device_g, audio_dev_on_device_g, chunk_idx)
    slots = [None] * S
    completed = []

    g_master = torch.Generator(device=devices[0]).manual_seed(args.seed)

    def make_new_chunk(chunk_idx):
        """Build fresh noisy chunk + audio for the input to GPU 0."""
        d = devices[0]
        a0 = chunk_idx * F_CHUNK * 4
        audio_chunk = audio_full[a0:a0 + T_AUDIO_CHUNK]
        if audio_chunk.shape[0] < T_AUDIO_CHUNK:
            audio_chunk = F.pad(audio_chunk, (0, 0, 0, T_AUDIO_CHUNK - audio_chunk.shape[0]))
        audio_dev = audio_chunk.unsqueeze(0).to(d, torch.bfloat16)
        noisy = torch.randn(1, 16, F_CHUNK, H, W, generator=g_master,
                            device=d, dtype=torch.bfloat16)
        noisy[:, :, :args.motion_latent_frames] = motion_per_gpu[0]
        return (noisy, audio_dev, chunk_idx)

    # Warmup + steady-state + drain: total ticks = num_chunks + S - 1
    total_ticks = args.num_chunks + S - 1
    next_chunk_idx = 0

    print(f"\nrunning pipeline: {args.num_chunks} chunks, {S} GPUs, "
          f"{total_ticks} ticks total")
    t_start = time.time()
    tick_times = []
    for tick in range(total_ticks):
        t_tick = time.time()
        new_slots = [None] * S

        # Each GPU runs its step on its current slot.
        # Streams launched concurrently across the 4 GPUs.
        streams = [torch.cuda.Stream(device=g) for g in gpu_ids]
        for g in range(S):
            if slots[g] is None:
                continue
            with torch.cuda.device(gpu_ids[g]), torch.cuda.stream(streams[g]):
                noisy, audio_dev, chunk_idx = slots[g]
                next_noisy, clean = run_step(g, noisy, audio_dev)

                if g < S - 1:
                    # Move output + audio to next GPU's device
                    d_next = devices[g + 1]
                    nn = next_noisy.to(d_next, non_blocking=True)
                    ad = audio_dev.to(d_next, non_blocking=True)
                    new_slots[g + 1] = (nn, ad, chunk_idx)
                else:
                    completed.append((clean.cpu(), chunk_idx))

        # Inject a new chunk at slot 0 if any chunks left to enqueue
        if next_chunk_idx < args.num_chunks:
            new_slots[0] = make_new_chunk(next_chunk_idx)
            next_chunk_idx += 1

        # Wait for all GPUs' streams to finish this tick
        for g in range(S):
            torch.cuda.synchronize(gpu_ids[g])

        slots = new_slots
        tick_dt = time.time() - t_tick
        tick_times.append(tick_dt)
        n_busy = sum(1 for s in slots if s is not None)
        print(f"  tick {tick:2d}  dt={tick_dt*1000:.0f}ms  "
              f"completed={len(completed)}  busy_slots={n_busy}", flush=True)

    total = time.time() - t_start
    avg_tick = sum(tick_times) / len(tick_times)
    steady_tick = sum(tick_times[S-1:]) / max(1, len(tick_times) - (S-1))
    print(f"\nDONE: {args.num_chunks} chunks in {total:.2f}s ")
    print(f"  avg tick:     {avg_tick*1000:.0f}ms")
    print(f"  steady tick:  {steady_tick*1000:.0f}ms  (= 1 chunk per tick after warmup)")
    print(f"  throughput:   {args.num_chunks/total:.2f} chunks/s "
          f"= {args.num_chunks * F_CHUNK * 4 / 25 / total:.2f}x real-time "
          f"(target >1.0)")

    # ---- Decode + save -----------------------------------------------------
    print(f"\ndecoding {len(completed)} chunks on {devices[-1]}...")
    for clean_cpu, chunk_idx in completed:
        clean = clean_cpu.to(devices[-1], dtype=torch.bfloat16)
        pix = decode_latent(vae, clean.squeeze(0), devices[-1])
        F_pix = pix.shape[1]
        for fl, f_idx in [("first", 0), ("mid", F_pix // 2), ("last", F_pix - 1)]:
            img = ((pix[:, f_idx] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(
                os.path.join(args.out_dir, f"chunk{chunk_idx:02d}_{fl}.png"))
    print(f"saved {len(completed)*3} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
