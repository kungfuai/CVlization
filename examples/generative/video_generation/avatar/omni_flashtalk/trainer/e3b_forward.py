"""E3b: forward with audio injection. Validate it runs + check stats."""
import sys
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
import torch
from omni_adapter_v2 import OmniAudioWanModel, load_omni_into_adapter

model = OmniAudioWanModel(
    model_type='t2v', patch_size=(1, 2, 2), text_len=512,
    in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=12, num_layers=30, window_size=(-1, -1),
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
)
model = load_omni_into_adapter(
    model,
    "/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt",
)
model = model.to(dtype=torch.bfloat16, device="cuda").eval()

# Build inputs sized so that audio time-rate aligns with video latent frames.
# audio_emb shape (B, T_aud_frames, 10752); after AudioPack [4,1,1] +3-frame
# pad, T_aud_packed = (T_aud_frames + 3) / 4. Must equal F_patch (==F_latent
# since patch temporal stride is 1).
B = 1
F_lat = 21
H_lat = 60     # 480 / 8
W_lat = 104    # 832 / 8
# T_aud_packed must equal F_lat=21  =>  T_aud_frames = 21*4 - 3 = 81
T_aud_frames = 81
audio_input_dim = 10752

# === Forward WITHOUT audio (sanity) ===
with torch.no_grad():
    x = [torch.randn(33, F_lat, H_lat, W_lat, dtype=torch.bfloat16, device="cuda")]
    context = [torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")]
    t = torch.tensor([500.], dtype=torch.bfloat16, device="cuda")
    seq_len = F_lat * H_lat * W_lat // (1 * 2 * 2)
    out_no_audio = model(x, t, context, seq_len, audio_emb=None)
    print(f"  out_no_audio shape: {tuple(out_no_audio.shape)}  mean={out_no_audio.mean():.4e}  std={out_no_audio.std():.4e}")

# === Forward WITH audio ===
with torch.no_grad():
    audio_emb = torch.randn(B, T_aud_frames, audio_input_dim, dtype=torch.bfloat16, device="cuda")
    print(f"  GPU mem before audio fwd: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    out_audio = model(x, t, context, seq_len, audio_emb=audio_emb)
    print(f"  GPU mem after audio fwd:  {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  out_audio shape: {tuple(out_audio.shape)}  mean={out_audio.mean():.4e}  std={out_audio.std():.4e}")

# === Difference ===
diff = (out_audio - out_no_audio).float()
print(f"  diff (audio - no_audio): mean={diff.mean():.4e}  std={diff.std():.4e}  max_abs={diff.abs().max():.4e}")
print(f"  relative magnitude of audio contribution: {diff.std()/out_no_audio.float().std():.4f}")
print("  E3b FORWARD OK")
