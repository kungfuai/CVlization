"""E4: validate causal forward with audio injection."""
import sys
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
import torch
from omni_causal_adapter import OmniAudioCausalWanModel, load_omni_into_causal_adapter

torch.manual_seed(42)

model = OmniAudioCausalWanModel(
    model_type='t2v', patch_size=(1, 2, 2), text_len=512,
    in_dim=33, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=12, num_layers=30, local_attn_size=-1, sink_size=0,
    qk_norm=True, cross_attn_norm=True, eps=1e-6,
)
# CausalWanModel sets num_frame_per_block=1 by default; pick 3 to match
# Self-Forcing's published recipe.
model.num_frame_per_block = 3
model = load_omni_into_causal_adapter(
    model,
    "/home/whadmin/zz/omni_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/home/whadmin/zz/omni_models/OmniAvatar-1.3B/pytorch_model.pt",
)
model = model.to(dtype=torch.bfloat16, device="cuda").eval()

# Smaller dims for the smoke (block-causal mask construction can be heavy)
B, F_lat, H_lat, W_lat = 1, 21, 30, 52
seq_len = F_lat * H_lat * W_lat // (1 * 2 * 2)
print(f"  smoke dims: F={F_lat} H={H_lat} W={W_lat} seq_len={seq_len}")
print(f"  num_frame_per_block={model.num_frame_per_block}, local_attn_size={model.local_attn_size}")

with torch.no_grad():
    x = [torch.randn(33, F_lat, H_lat, W_lat, dtype=torch.bfloat16, device="cuda")]
    context = [torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")]
    t = torch.full((1, F_lat), 500., dtype=torch.bfloat16, device="cuda")
    audio_emb = torch.randn(B, 81, 10752, dtype=torch.bfloat16, device="cuda")

    print(f"  GPU mem before forward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    out = model(x, t, context, seq_len, audio_emb=audio_emb)
    print(f"  GPU mem after forward:  {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  output shape: {tuple(out.shape)}, dtype: {out.dtype}")
    print(f"  output stats: mean={out.float().mean():.4e}  std={out.float().std():.4e}")

print("  E4 CAUSAL FORWARD: OK")
