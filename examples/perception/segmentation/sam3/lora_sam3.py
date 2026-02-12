"""LoRA adapters for SAM3's ViT backbone attention blocks.

Injects low-rank adapters into the fused QKV projections of the
ViT-L/14 backbone (32 blocks, dim=1024).  Only the LoRA A/B matrices
are trainable; everything else is frozen.

Adapted from sam_lora_finetuning/lora.py for SAM3's architecture.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger(__name__)


class LoRA_qkv(nn.Module):
    """LoRA adaption injected into a fused QKV attention linear layer.

    Adds low-rank updates to the Q and V projections while leaving K
    unchanged.  Handles both 3-D ``(B, L, C)`` and 4-D ``(B, H, W, C)``
    input tensors (SAM3's ViT uses 4-D for spatial blocks).
    """

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # (..., 3*dim)
        q_ba = self.linear_b_q(self.linear_a_q(x))  # (..., dim)
        v_ba = self.linear_b_v(self.linear_a_v(x))  # (..., dim)
        qkv[..., : self.d_model] += q_ba
        qkv[..., -self.d_model :] += v_ba
        return qkv


def inject_lora(model, rank=64, lora_layer=None):
    """Inject LoRA adapters into the SAM3 ViT backbone attention blocks.

    The ViT backbone lives at ``model.backbone.vision_backbone.trunk``.

    Args:
        model: SAM3 image model (``Sam3Image``).
        rank: LoRA rank for the low-rank decomposition.
        lora_layer: List of block indices to inject LoRA into.
            ``None`` means all blocks.

    Returns:
        Tuple ``(A_weights, B_weights)`` — lists of the LoRA weight
        modules, useful for saving / loading.
    """
    vit = model.backbone.vision_backbone.trunk
    blocks = vit.blocks

    if lora_layer is None:
        lora_layer = list(range(len(blocks)))

    A_weights = []
    B_weights = []

    for i, blk in enumerate(blocks):
        if i not in lora_layer:
            continue

        w_qkv = blk.attn.qkv
        d_model = w_qkv.in_features

        w_a_q = nn.Linear(d_model, rank, bias=False)
        w_b_q = nn.Linear(rank, d_model, bias=False)
        w_a_v = nn.Linear(d_model, rank, bias=False)
        w_b_v = nn.Linear(rank, d_model, bias=False)

        A_weights.extend([w_a_q, w_a_v])
        B_weights.extend([w_b_q, w_b_v])

        blk.attn.qkv = LoRA_qkv(w_qkv, w_a_q, w_b_q, w_a_v, w_b_v)

    # Initialise per the LoRA paper: Kaiming for A, zeros for B.
    for w_A in A_weights:
        nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
    for w_B in B_weights:
        nn.init.zeros_(w_B.weight)

    log.info(
        "Injected LoRA (rank=%d) into %d ViT blocks (%d A + %d B matrices)",
        rank,
        len(lora_layer),
        len(A_weights),
        len(B_weights),
    )
    return A_weights, B_weights


def save_lora_parameters(A_weights, B_weights, filename):
    """Save LoRA A/B weight matrices as a safetensors file."""
    tensors = {}
    for i, w in enumerate(A_weights):
        tensors[f"w_a_{i:03d}"] = w.weight
    for i, w in enumerate(B_weights):
        tensors[f"w_b_{i:03d}"] = w.weight
    save_file(tensors, filename)
    log.info("Saved %d LoRA tensors to %s", len(tensors), filename)


def load_lora_parameters(A_weights, B_weights, filename):
    """Load LoRA A/B weight matrices from a safetensors file."""
    with safe_open(filename, framework="pt") as f:
        for i, w_A in enumerate(A_weights):
            w_A.weight = nn.Parameter(f.get_tensor(f"w_a_{i:03d}"))
        for i, w_B in enumerate(B_weights):
            w_B.weight = nn.Parameter(f.get_tensor(f"w_b_{i:03d}"))
    log.info("Loaded LoRA tensors from %s", filename)


# ---------------------------------------------------------------------------
# Hydra-compatible model builder
# ---------------------------------------------------------------------------


def build_sam3_lora_model(
    lora_rank=64,
    unfreeze_decoder=False,
    unfreeze_seg_head=False,
    **kwargs,
):
    """Build a SAM3 image model with LoRA adapters in the ViT backbone.

    This function is designed to be used as a Hydra ``_target_`` in a
    training config.  It accepts the same keyword arguments as
    :func:`sam3.model_builder.build_sam3_image_model` plus LoRA-specific
    parameters.

    Args:
        lora_rank: LoRA rank for QKV attention projections.
        unfreeze_decoder: If ``True``, unfreeze the transformer decoder.
        unfreeze_seg_head: If ``True``, unfreeze the segmentation head
            (pixel decoder + mask predictor).
        **kwargs: Forwarded to ``build_sam3_image_model``.

    Returns:
        A ``Sam3Image`` model with LoRA injected and parameters frozen
        according to the configuration.
    """
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(**kwargs)

    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Inject LoRA into ViT backbone (new params are requires_grad=True)
    A_weights, B_weights = inject_lora(model, rank=lora_rank)

    # 3. Optionally unfreeze decoder
    if unfreeze_decoder:
        for param in model.transformer.decoder.parameters():
            param.requires_grad = True
        log.info("Unfroze transformer decoder")

    # 4. Optionally unfreeze segmentation head
    if unfreeze_seg_head and model.segmentation_head is not None:
        for param in model.segmentation_head.parameters():
            param.requires_grad = True
        log.info("Unfroze segmentation head")

    # Log trainable stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "Trainable params: %s / %s (%.1f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # Stash LoRA weight references on the model for later saving
    model._lora_A_weights = A_weights
    model._lora_B_weights = B_weights

    return model
