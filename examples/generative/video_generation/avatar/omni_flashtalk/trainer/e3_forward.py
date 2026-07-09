"""E3 forward-pass test: validate the ported model runs end-to-end on GPU.

We don't yet thread audio_emb through the block iteration (that's a forward-side
patch, separate from the module-construction patch tested here). What this
proves: the audio modules + 33-channel patch_embedding integrate cleanly with
the SF WanModel forward path, and a bidirectional forward call produces a
sensibly-shaped output.
"""
import sys
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
import torch
from omni_adapter import OmniAudioWanModel, load_omni_into_adapter

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

# SF forward signature: x is List[Tensor[C,F,H,W]], context is List[Tensor[L,C]], t is Tensor[B]
B = 1
F_lat = 21       # frames in latent space
H_lat = 60       # 480 / 8
W_lat = 104      # 832 / 8
C_in = 33        # OmniAvatar's expanded in_channels
L_text = 512
C_text = 4096

with torch.no_grad():
    x = [torch.randn(C_in, F_lat, H_lat, W_lat, dtype=torch.bfloat16, device="cuda")]
    context = [torch.randn(L_text, C_text, dtype=torch.bfloat16, device="cuda")]
    t = torch.tensor([500.], dtype=torch.bfloat16, device="cuda")
    seq_len = F_lat * H_lat * W_lat // (1 * 2 * 2)  # patch_size=(1,2,2)
    print(f"  input x[0] shape: {tuple(x[0].shape)}")
    print(f"  context[0] shape: {tuple(context[0].shape)}")
    print(f"  seq_len: {seq_len}")

    print(f"  GPU mem before forward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    out = model(x, t, context, seq_len)
    print(f"  GPU mem after forward:  {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"  output type: {type(out).__name__}, len: {len(out) if isinstance(out, list) else 'n/a'}")
    if isinstance(out, list):
        print(f"  output[0] shape: {tuple(out[0].shape)}, dtype: {out[0].dtype}")
        print(f"  output[0] stats: mean={out[0].mean().item():.4e}  std={out[0].std().item():.4e}")
print("  FORWARD OK")
