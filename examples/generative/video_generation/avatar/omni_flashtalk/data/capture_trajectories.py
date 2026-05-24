"""Re-run the SoulX-FlashTalk teacher with sample_steps=4 and snapshot the
latent at each of the 4 timesteps + the final clean x0. Saves a 5-step ODE
trajectory per item to `trajectories/<id>.pt`.

Why: per Hallo-Live / CausVid / Self-Forcing, KD on a pretrained student needs
the teacher's *actual* intermediate latents (not a fresh x0+random-noise mix),
otherwise the student sees out-of-distribution inputs and the gradient blows
up. SoulX already runs with `sample_steps=4` matching Hallo-Live's
`denoising_step_list=[1000, 750, 500, 250]` — we just need to keep the
intermediates this time instead of throwing them away after decode.

Run on a host with SoulX-FlashTalk + its venv installed (no Docker needed):
    /home/whadmin/zz/SoulX-FlashTalk/.venv/bin/python capture_trajectories.py \\
        --data-dir ~/zz/omni_flashtalk_data \\
        --soulx-root ~/zz/SoulX-FlashTalk \\
        --ckpt-dir ~/zz/soulx_models/SoulX-FlashTalk \\
        --wav2vec-dir ~/zz/soulx_models/wav2vec2-base-960h \\
        --gpu 0
"""
import argparse
import json
import os
import sys
import time

import librosa
import numpy as np
import torch
import yaml


def make_pipeline(args):
    sys.path.insert(0, args.soulx_root)
    from flash_talk.src.pipeline.flash_talk_pipeline import FlashTalkPipeline
    from flash_talk.src.distributed.usp_device import get_device, get_parallel_degree
    from flash_talk.infinite_talk.configs import multitalk_14B

    cfg = multitalk_14B
    ud, rd = get_parallel_degree(1, cfg.num_heads)
    device = get_device(ud, rd)
    # cpu_offload=True is required to fit the 14B teacher on a single 40GB GPU
    # (the existing 100 KD targets were generated this way too)
    pipe = FlashTalkPipeline(
        config=cfg, checkpoint_dir=args.ckpt_dir, wav2vec_dir=args.wav2vec_dir,
        device=device, use_usp=False, cpu_offload=args.cpu_offload,
    )

    # ---- patch generate() to snapshot trajectory ---------------------------
    # Body is the SoulX denoising loop verbatim (flash_talk_pipeline.py L274+)
    # with three additions:
    #   1. trajectory list = clone latent at the START of each iteration
    #   2. after the loop, append the final clean x_0 (which equals `latent`
    #      since the last timestep is 0 -> latent = (1-0)*x_0 + 0*noise)
    #   3. return (decoded_video, trajectory) instead of just video
    @torch.no_grad()
    def generate_with_traj(self, audio_embedding):
        if self.cpu_offload:
            self.model.to(self.device)
        self.arg_c.update({"audio": audio_embedding})

        latent = torch.randn(
            16, (self.frame_num - 1) // 4 + 1, self.lat_h, self.lat_w,
            dtype=self.param_dtype, device=self.device, generator=self.generator,
        )
        latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames

        trajectory = []
        for i in range(len(self.timesteps) - 1):
            trajectory.append(latent.detach().to(torch.bfloat16).cpu().clone())
            timestep = self.timesteps[i]
            noise_pred_cond = self.model([latent], t=timestep, **self.arg_c)[0]
            noise_pred = -noise_pred_cond
            t_i = self.timesteps[i][:, None, None, None] / self.num_timesteps
            t_i_1 = self.timesteps[i + 1][:, None, None, None] / self.num_timesteps
            x_0 = latent + noise_pred * t_i
            latent = (1 - t_i_1) * x_0 + t_i_1 * torch.randn(
                x_0.size(), dtype=x_0.dtype, device=self.device, generator=self.generator)
            latent[:, :self.latent_motion_frames.shape[1]] = self.latent_motion_frames
        # last timestep is 0 -> latent now equals x_0 (clean)
        trajectory.append(latent.detach().to(torch.bfloat16).cpu().clone())

        # Restore cpu_offload state before VAE decode (which expects GPU VAE).
        if self.cpu_offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.vae.model.to(self.device)
        videos = self.vae.decode(latent.to(self.param_dtype))
        from flash_talk.src.pipeline.flash_talk_pipeline import match_and_blend_colors_torch
        if self.color_correction_strength > 0.0:
            videos = match_and_blend_colors_torch(
                videos, self.original_color_reference, self.color_correction_strength)
        # refresh motion-frame anchor for the next call
        cond_frame = videos[:, :, -self.motion_frames_num:].to(self.device)
        self.latent_motion_frames = self.vae.encode(cond_frame)
        if self.cpu_offload:
            self.vae.model.cpu()
            torch.cuda.empty_cache()

        return videos[0].to(torch.float32), torch.stack(trajectory, dim=0)

    import types
    pipe.generate = types.MethodType(generate_with_traj, pipe)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir holding manifest_assets.jsonl + audio/ + portraits/")
    ap.add_argument("--soulx-root", required=True,
                    help="path to SoulX-FlashTalk source")
    ap.add_argument("--ckpt-dir", required=True,
                    help="SoulX-FlashTalk model checkpoint dir")
    ap.add_argument("--wav2vec-dir", required=True,
                    help="wav2vec2-base-960h dir")
    ap.add_argument("--params-yaml",
                    default="flash_talk/configs/infer_params.yaml",
                    help="relative to --soulx-root")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="offload model to CPU between steps; required for 14B on 40GB")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N items (0 = all)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    data_dir = os.path.expanduser(args.data_dir)
    out_dir = os.path.join(data_dir, "trajectories")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.soulx_root, args.params_yaml)) as f:
        params = yaml.safe_load(f)
    assert params["sample_steps"] == 4, (
        f"infer_params.yaml sample_steps must be 4 to match Hallo-Live recipe; "
        f"got {params['sample_steps']}")

    print(f"loading SoulX (this takes ~30s)...")
    pipe = make_pipeline(args)

    items = [json.loads(l) for l in
             open(os.path.join(data_dir, "manifest_assets.jsonl")) if l.strip()]
    if args.limit:
        items = items[:args.limit]

    done = skipped = 0
    t_start = time.time()
    for it in items:
        out_path = os.path.join(out_dir, f"{it['id']}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue

        t0 = time.time()
        audio, _ = librosa.load(
            os.path.join(data_dir, it["audio_path"]),
            sr=params["sample_rate"])
        cond_image = os.path.join(data_dir, it["image_path"])

        # Mirror what generate_video.py does: prepare_params, then preprocess
        # audio, then generate. We use the item's full_prompt as the text.
        pipe.prepare_params(
            input_prompt=it["full_prompt"],
            cond_image=cond_image,
            target_size=(params["height"], params["width"]),
            frame_num=params["frame_num"],
            motion_frames_num=params["motion_frames_num"],
            sampling_steps=params["sample_steps"],
            seed=42,
            shift=params["sample_shift"],
            color_correction_strength=params["color_correction_strength"],
        )
        audio_emb = pipe.preprocess_audio(
            audio, sr=params["sample_rate"], fps=params["tgt_fps"])

        # Index into audio_emb to span the whole video. SoulX's inference
        # builds a sliding window of 2*2+1 = 5 frames around each output frame.
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        ai_start, ai_end = 0, audio_emb.shape[0]
        centers = (torch.arange(ai_start, ai_end, 1).unsqueeze(1) +
                   indices.unsqueeze(0))
        centers = torch.clamp(centers, min=0, max=ai_end - 1)
        # SoulX generate() expects 5D: [1, T_aud_chunk, window=5, n_layers, D]
        # where T_aud_chunk == frame_num (one chunk = 33 video frames). The full
        # video is normally produced by looping generate() over chunks; for KD
        # trajectory capture we only need ONE chunk's trajectory per item.
        full = audio_emb[centers][None, ...].contiguous().to(pipe.device)
        audio_emb_window = full[:, :params["frame_num"]]   # [1, 33, 5, 12, 768]

        try:
            video, traj = pipe.generate(audio_emb_window)
        except Exception as e:
            print(f"[{it['id']}] FAIL: {e}", flush=True)
            continue

        # traj: [5, C, F, H, W]  (4 noisy intermediates @ t=1000/750/500/250 + clean x0)
        # video: float32 pixel tensor we discard (we already have soulx_targets/*.mp4)
        torch.save({
            "ode_traj": traj,
            "denoising_steps": [1000, 750, 500, 250, 0],
            "sample_shift": params["sample_shift"],
        }, out_path)

        done += 1
        dt = time.time() - t0
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1.0)
        eta = (len(items) - done - skipped) / max(rate, 1e-6)
        print(f"[{done + skipped:3d}/{len(items)}] {it['id']}  "
              f"traj{tuple(traj.shape)}  {dt:.1f}s  eta={eta/60:.0f}m",
              flush=True)

    print(f"done: {done} new + {skipped} skipped in {(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
