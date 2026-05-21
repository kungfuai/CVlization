"""E5 Stage-1 KD smoke test (with gradient checkpointing + smaller dims)."""
import sys, time
sys.path.insert(0, "/home/whadmin/zz/Self-Forcing")
sys.path.insert(0, "/home/whadmin/zz/OmniAvatar")
import torch
import torch.nn.functional as F
from omni_adapter_v2 import OmniAudioWanModel, load_omni_into_adapter

torch.manual_seed(42)

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
model = model.to(dtype=torch.bfloat16, device="cuda").train()
# Enable grad ckpt via our adapter's check
model.gradient_checkpointing = True
print(f"  gradient_checkpointing: {model.gradient_checkpointing}")

# Trainable subset: audio + LoRA + patch_embedding only
for n, p in model.named_parameters():
    p.requires_grad_("lora_" in n or "audio_" in n or "patch_embedding" in n)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"  trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M ({100*n_trainable/n_total:.2f}%)")

# Smaller spatial for smoke: 240x416 latent (half res)
B, F_lat, H_lat, W_lat = 1, 21, 30, 52
seq_len = F_lat * H_lat * W_lat // (1 * 2 * 2)  # 8190
print(f"  smoke dims: F={F_lat} H={H_lat} W={W_lat} seq_len={seq_len}")

DATASET = []
for i in range(3):
    DATASET.append({
        "x":       [torch.randn(33, F_lat, H_lat, W_lat, dtype=torch.bfloat16, device="cuda")],
        "context": [torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")],
        "t":       torch.tensor([100 + 200 * i], dtype=torch.bfloat16, device="cuda"),
        "audio":   torch.randn(B, 81, 10752, dtype=torch.bfloat16, device="cuda"),
        "target":  torch.randn(B, seq_len, 64, dtype=torch.bfloat16, device="cuda"),
    })

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0,
)
print(f"  GPU mem after optim: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

N_STEPS = 50
losses = []
t0 = time.time()
for step in range(N_STEPS):
    item = DATASET[step % 3]
    pred = model(item["x"], item["t"], item["context"], seq_len, audio_emb=item["audio"])
    loss = F.mse_loss(pred.float(), item["target"].float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    opt.step()
    losses.append(loss.item())
    if step in (0,1,2,3,4) or step % 10 == 0 or step == N_STEPS - 1:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  step {step:3d}  loss={loss.item():.4f}  peak_mem={peak:.2f}GB")

elapsed = time.time() - t0
print(f"  trained {N_STEPS} steps in {elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step)")
print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  drop={(losses[0]-losses[-1])/losses[0]*100:.1f}%")
print(f"  has_nan={any(l!=l for l in losses)}")
print("  E5 SMOKE: PASS" if losses[-1] < losses[0]*0.5 else "  E5 SMOKE: loss did not drop > 50%")
