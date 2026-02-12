"""LoRA adapters for SAM's image encoder attention blocks.

Ported from https://github.com/JamesQFreeman/Sam_LoRA — adapted to use
the `segment_anything` pip package instead of vendored source.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.sam import Sam


class LoRA_qkv(nn.Module):
    """LoRA adaption injected into a QKV attention linear layer.

    Adds low-rank updates to the Q and V projections while leaving K unchanged.
    """

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.d_model] += q_ba
        qkv[:, :, :, -self.d_model :] += v_ba
        return qkv


class LoRA_sam(nn.Module):
    """Wraps a SAM model, injecting LoRA adapters into the image encoder.

    All image-encoder parameters are frozen; only the LoRA A/B matrices are
    trainable.  Prompt encoder and mask decoder remain trainable by default.
    """

    def __init__(self, sam_model: Sam, rank: int, lora_layer=None):
        super().__init__()
        assert rank > 0
        self.rank = rank

        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.A_weights = []
        self.B_weights = []

        # Freeze image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(d_model, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, d_model, bias=False)
            w_a_linear_v = nn.Linear(d_model, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, d_model, bias=False)

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder

    def reset_parameters(self):
        """Kaiming init for A matrices, zeros for B (per LoRA paper)."""
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str):
        """Save LoRA weights as safetensors."""
        num_layer = len(self.A_weights)
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        save_file({**a_tensors, **b_tensors}, filename)

    def load_lora_parameters(self, filename: str):
        """Load LoRA weights from safetensors."""
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_tensor = f.get_tensor(f"w_a_{i:03d}")
                w_A_linear.weight = nn.Parameter(saved_tensor)
            for i, w_B_linear in enumerate(self.B_weights):
                saved_tensor = f.get_tensor(f"w_b_{i:03d}")
                w_B_linear.weight = nn.Parameter(saved_tensor)
