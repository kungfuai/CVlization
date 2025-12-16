# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
import torch.utils.checkpoint

from .attention import attention
from .model import (
    Head,
    WanAttentionBlock,
    WanRMSNorm,
    WanLayerNorm,
    WanModel,
    WanSelfAttention,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
)
from .model_s2v import (
    FramePackMotioner,
    rope_apply,
    rope_precompute,
)
from .inference_utils import conditional_compile, disable, NO_REFRESH_INFERENCE

from .audio_utils import CausalAudioEncoder, AudioInjector_WAN

from ...utils import parallel_state as mpu
from ...utils.all_to_all import SeqAllToAll4D

flex_attention = torch.compile(
    flex_attention, mode="default", dynamic=True)

@conditional_compile
def rope_t_params(t, num_frame_per_block, dim, theta=10000, device='cpu'):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(0, num_frame_per_block, device=device) + t,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@conditional_compile
def get_rope_t_params(t, num_frame_per_block, d, device):
    freqs = rope_t_params(t, num_frame_per_block, d - 4 * (d // 6), device=device)
    return freqs

@conditional_compile
def causal_rope_apply_t_only(x, grid_sizes, freqs_i):
    n, c = x.size(2), x.size(3) // 2

    output = []

    i = 0
    f, h, w = grid_sizes
    seq_len = f * h * w

    x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
        seq_len, n, -1, 2))

    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[i, seq_len:]])

    # append to collection
    output.append(x_i)

    return torch.stack(output).type_as(x)

@conditional_compile
def causal_rope_apply(x, grid_sizes, freqs_i=None):
    n, c = x.size(2), x.size(3) // 2

    output = []
    i = 0
    f, h, w = grid_sizes
    seq_len = f * h * w

    x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
        seq_len, n, -1, 2))

    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[i, seq_len:]])

    output.append(x_i)
    return torch.stack(output).type_as(x)

@conditional_compile
def precompute_freqs_i(grid_sizes, freqs_t, freqs_hw, head_dim, start_frame=0, sp_dim=None):
    c = head_dim // 2
    sp_size = mpu.get_sequence_parallel_world_size()
    sp_rank = mpu.get_sequence_parallel_rank()

    freqs_hw = freqs_hw.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_t = [freqs_t]

    f, h, w = grid_sizes
    seq_len = f * h * w
    if sp_size > 1 and sp_dim == 'h':
            freqs_i = torch.cat([
            freqs_t[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_hw[1][h*sp_rank:h*(sp_rank+1)].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_hw[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
    elif sp_size > 1 and sp_dim == 'w':
        freqs_i = torch.cat([
            freqs_t[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_hw[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_hw[2][w*sp_rank:w*(sp_rank+1)].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
    else:
        assert mpu.get_sequence_parallel_world_size() == 1
        freqs_i = torch.cat([
            freqs_t[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_hw[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_hw[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)
    return freqs_i

@conditional_compile
def precompute_freqs_i_t_only(grid_sizes, freqs_t, freqs_hw, head_dim, start_frame=0):
    c = head_dim // 2

    freqs_t = [freqs_t]
    freqs_hw = freqs_hw.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w

    freqs_i = torch.cat([
        freqs_t[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_hw[1][0:1].expand(h, -1).view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_hw[2][0:1].expand(w, -1).view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    return freqs_i

def convert_grid_sizes(grid_sizes, current_start):
    if isinstance(grid_sizes, list):
        grid_sizes = grid_sizes[0][1]
    grid_sizes = grid_sizes.flatten().tolist()[:3]
    if current_start == 0:
        grid_sizes[0] = 1
    return grid_sizes


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class CausalWanS2VSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.sp_size = mpu.get_sequence_parallel_world_size()
        self.sp_group = mpu.get_sequence_parallel_group()
        self.scatter_idx = 2
        self.gather_idx = 1

        if NO_REFRESH_INFERENCE:
            self.freqs_t_1 = get_rope_t_params(1, 1, self.head_dim, device='cuda')
            self.freqs_i_1_size_cache = None
            self.freqs_i_1_cache = None

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None, frame_seqlen=None, sp_dim=None, sink_size=None, freqs_i=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            frame_seqlen(int): Frame sequence length after sp split
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs, sp_dim=sp_dim).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs, sp_dim=sp_dim).type_as(v)

            if self.sp_size > 1:
                gqa_backward_allreduce = False
                roped_query = SeqAllToAll4D.apply(self.sp_group, roped_query, self.scatter_idx, self.gather_idx, False)
                roped_key = SeqAllToAll4D.apply(self.sp_group, roped_key, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)
                v = SeqAllToAll4D.apply(self.sp_group, v, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)

            padded_length = math.ceil(roped_query.shape[1] / 128) * 128 - roped_query.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([roped_query.shape[0], padded_length, roped_query.shape[2], roped_query.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([roped_key.shape[0], padded_length, roped_key.shape[2], roped_key.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, : roped_query.shape[1]].transpose(2, 1)

            if self.sp_size > 1:
                x = SeqAllToAll4D.apply(self.sp_group, x, self.gather_idx, self.scatter_idx, False)

        else:
            if sink_size is None:
                sink_size = 0

            sink_tokens = round(sink_size * frame_seqlen)
            max_attention_size = frame_seqlen * (20 if self.local_attn_size == -1 else self.local_attn_size * self.num_frame_per_block)
            current_start_value = (current_start if not isinstance(current_start, torch.Tensor) else current_start.item())
            current_frame = current_start_value // frame_seqlen
            current_end = current_start_value + q.shape[1]

            if NO_REFRESH_INFERENCE:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs_i=freqs_i)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs_i=freqs_i)

                sink_grid_size = [1] + grid_sizes[1:]
                if sink_grid_size != self.freqs_i_1_size_cache:
                    self.freqs_i_1_size_cache = sink_grid_size
                    self.freqs_i_1_cache = precompute_freqs_i_t_only(sink_grid_size, self.freqs_t_1, freqs, self.head_dim)

                if current_end > kv_cache['global_end_index'].item() and current_frame > 20:
                    # update rope of sink_tokens
                    kv_cache['k'][:, :sink_tokens] = causal_rope_apply(kv_cache['k'][:, :sink_tokens], sink_grid_size, freqs_i=self.freqs_i_1_cache)

            else:
                roped_query = rope_apply(
                    q, grid_sizes, freqs, sp_dim=sp_dim, current_start=current_start).type_as(v)
                roped_key = rope_apply(
                    k, grid_sizes, freqs, sp_dim=sp_dim, current_start=current_start).type_as(v)

            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            rolling_cache_size = kv_cache_size - sink_tokens

            if current_start_value < sink_tokens: # prefill
                local_start_index = current_start_value
                local_end_index = current_end
                local_end_rounds = 0
            else:
                local_start_index = (current_start_value - sink_tokens) % rolling_cache_size + sink_tokens
                local_end_index = (current_end - sink_tokens) % rolling_cache_size + sink_tokens
                local_end_rounds = (current_end - sink_tokens) // rolling_cache_size

            if local_end_index > local_start_index:
                kv_cache['k'][:, local_start_index: local_end_index] = roped_key
                kv_cache['v'][:, local_start_index: local_end_index] = v
            else:
                slice_size = kv_cache_size - local_start_index
                assert 0 <= slice_size <= roped_key.shape[1]

                kv_cache['k'][:, local_start_index: ] = roped_key[:, : slice_size]
                kv_cache['v'][:, local_start_index: ] = v[:, : slice_size]
                kv_cache['k'][:, sink_tokens: local_end_index] = roped_key[:, slice_size: ]
                kv_cache['v'][:, sink_tokens: local_end_index] = v[:, slice_size: ]

            if (kv_cache_size == sink_tokens + max_attention_size) and current_end >= kv_cache_size:
                k = kv_cache['k']
                v = kv_cache['v']

            else:
                attn_start_index = (max(current_end - max_attention_size, sink_tokens) - sink_tokens) % rolling_cache_size + sink_tokens
                attn_start_rounds = (max(current_end - max_attention_size, sink_tokens) - sink_tokens) // rolling_cache_size
                if attn_start_rounds == local_end_rounds:
                    if attn_start_index == sink_tokens:
                        k = kv_cache['k'][:, : local_end_index]
                        v = kv_cache['v'][:, : local_end_index]
                    else:
                        k = torch.cat([kv_cache['k'][:, : sink_tokens], kv_cache['k'][:, attn_start_index: local_end_index]], dim=1)
                        v = torch.cat([kv_cache['v'][:, : sink_tokens], kv_cache['v'][:, attn_start_index: local_end_index]], dim=1)

                elif attn_start_rounds < local_end_rounds:
                    assert attn_start_rounds + 1 == local_end_rounds, f'attn_start_rounds, local_end_rounds: {attn_start_rounds}, {local_end_rounds}, {attn_start_index}, {local_end_index}'
                    k = torch.cat([kv_cache['k'][:, : local_end_index], kv_cache['k'][:, attn_start_index: ]], dim=1)
                    v = torch.cat([kv_cache['v'][:, : local_end_index], kv_cache['v'][:, attn_start_index: ]], dim=1)

                else:
                    raise NotImplementedError

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

            if self.sp_size > 1:
                gqa_backward_allreduce = False
                roped_query = SeqAllToAll4D.apply(self.sp_group, roped_query, self.scatter_idx, self.gather_idx, False)
                k = SeqAllToAll4D.apply(self.sp_group, k, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)
                v = SeqAllToAll4D.apply(self.sp_group, v, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)

            x = attention(
                roped_query,
                k,
                v
            )

            if self.sp_size > 1:
                x = SeqAllToAll4D.apply(self.sp_group, x, self.gather_idx, self.scatter_idx, False)

            # output
            x = x.flatten(2)
            x = self.o(x)
            return x


class CausalWanS2VAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_frame_per_block=1):
        super().__init__('t2v_cross_attn', dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps)
        self.local_attn_size = local_attn_size
        self.self_attn = CausalWanS2VSelfAttention(dim, num_heads, local_attn_size, sink_size,
                                             qk_norm, eps, num_frame_per_block=num_frame_per_block)


    @conditional_compile
    def forward(self, *args, **kwargs):
        if kwargs['mode'] == 'ode':
            return self.forward_ode(*args, **kwargs)
        elif kwargs['mode'] == 'dmd':
            return self.forward_dmd(*args, **kwargs)

    def forward_ode(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                block_mask,
                kv_cache=None,
                crossattn_cache=None,
                current_start=0,
                cache_start=None,
                frame_seqlen=None,
                sp_dim=None,
                sink_size=None,
                freqs_i=None,
                **kwargs
    ):
        # assert e[0].dtype == torch.float32
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, x.size(1) - seg_idx, x.size(1)]
        e = e[0] # B, T+1, 6, C
        num_frames, frame_seqlen = e.shape[1] - 1, (seg_idx[2] - seg_idx[1]) // (e.shape[1] - 1)
        modulation = self.modulation.unsqueeze(1) # 1,1,6,C
        # with amp.autocast(dtype=torch.float32):
        e = (modulation + e).chunk(6, dim=2) # list of [B, T+1, 1, C]
        # assert e[0].dtype == torch.float32

        norm_x = self.norm1(x).float()
        parts = []
        for i in range(2):
            scale = e[1][:, -1:] if i == 0 else e[1][:, :-1]
            shift = e[0][:, -1:] if i == 0 else e[0][:, :-1]
            seg_norm_x = norm_x[:, seg_idx[i]:seg_idx[i + 1]]
            if i == 1:
                seg_norm_x_ada = (seg_norm_x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)
            else:
                seg_norm_x_ada = seg_norm_x * (1 + scale.squeeze(2)) + shift.squeeze(2)

            parts.append(seg_norm_x_ada)

        norm_x = torch.cat(parts, dim=1)
        # self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs,
                block_mask, kv_cache, current_start, cache_start, frame_seqlen=frame_seqlen, sp_dim=sp_dim, sink_size=sink_size, freqs_i=freqs_i)
        # with amp.autocast(dtype=torch.float32):
        z = []
        for i in range(2):
            gate = e[2][:, -1:] if i == 0 else e[2][:, :-1]
            y_gated = y[:, seg_idx[i]:seg_idx[i + 1]]
            if i == 1:
                y_gated = (y_gated.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)
            else:
                y_gated = y_gated * gate.squeeze(2)
            z.append(y_gated)
        y = torch.cat(z, dim=1)
        x = x + y

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            norm2_x = self.norm2(x).float()
            parts = []
            for i in range(2):
                scale = e[4][:, -1:] if i == 0 else e[4][:, :-1]
                shift = e[3][:, -1:] if i == 0 else e[3][:, :-1]
                seg_norm2_x = norm2_x[:, seg_idx[i]:seg_idx[i + 1]]
                if i == 1:
                    seg_norm2_x_ada = (seg_norm2_x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)
                else:
                    seg_norm2_x_ada = seg_norm2_x * (1 + scale.squeeze(2)) + shift.squeeze(2)
                parts.append(seg_norm2_x_ada)

            norm2_x = torch.cat(parts, dim=1)
            y = self.ffn(norm2_x)
            # with amp.autocast(dtype=torch.float32):
            z = []
            for i in range(2):
                gate = e[5][:, -1:] if i == 0 else e[5][:, :-1]
                y_gated = y[:, seg_idx[i]:seg_idx[i + 1]]
                if i == 1:
                    y_gated = (y_gated.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)
                else:
                    y_gated = y_gated * gate.squeeze(2)
                z.append(y_gated)
            y = torch.cat(z, dim=1)
            x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x

    def forward_dmd(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                block_mask,
                kv_cache=None,
                crossattn_cache=None,
                current_start=0,
                cache_start=None,
                frame_seqlen=None,
                sp_dim=None,
                sink_size=None,
                freqs_i=None,
                **kwargs
    ):
        # assert e[0].dtype == torch.float32
        modulation = self.modulation # 1,6,C
        # with amp.autocast(dtype=torch.float32):
        e = (modulation + e).chunk(6, dim=1) # list of [B, 1, C]
        # assert e[0].dtype == torch.float32

        norm_x = self.norm1(x) * (1 + e[1]) + e[0]
        # self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs,
                block_mask, kv_cache, current_start, cache_start, frame_seqlen=frame_seqlen, sp_dim=sp_dim, sink_size=sink_size, freqs_i=freqs_i)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            norm2_x = self.norm2(x) * (1 + e[4]) + e[3]
            y = self.ffn(norm2_x)
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x

class CausalHead_S2V(Head):

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, T, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if e.ndim == 3:
            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2) # list of [B, T, 1, C]
            x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        else:
            assert e.ndim == 2
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1) # list of [B, 1, C]
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x

class CausalWanModel_S2V(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanS2VAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            cond_dim=0,
            audio_dim=5120,
            num_audio_token=4,
            enable_adain=False,
            adain_mode="attn_norm",
            audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
            zero_init=False,
            zero_timestep=False,
            add_last_motion=True,
            framepack_drop_mode="drop",
            model_type='s2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=2048,
            ffn_dim=8192,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=16,
            num_layers=32,
            local_attn_size=-1,
            sink_size=0,
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            num_frame_per_block=1,
            independent_first_frame=False,
            is_sparse=False,
            *args,
            **kwargs):
        super().__init__()

        assert model_type == 's2v'
        self.model_type = model_type
        self.is_sparse = is_sparse

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.frame_seqlen_cache = None
        self.motion_latents_seqlen_cache = 0
        self.max_latent_frames = 20

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanS2VAttentionBlock(dim, ffn_dim, num_heads, local_attn_size, sink_size, (-1, -1), qk_norm,
                                 cross_attn_norm, eps, num_frame_per_block=num_frame_per_block)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead_S2V(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # initialize weights
        self.init_weights()

        self.use_context_parallel = False  # will modify in _configure_model func

        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        self.enbale_adain = enable_adain

        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain)
        all_modules, all_modules_names = torch_dfs(
            self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
        )

        self.adain_mode = adain_mode
        self.trainable_cond_mask = nn.Embedding(3, self.dim)

        if zero_init:
            self.zero_init_weights()

        self.zero_timestep = zero_timestep  # Whether to assign 0 value timestep to ref/motion
        self.add_last_motion = add_last_motion

        self.frame_packer = FramePackMotioner(
            inner_dim=self.dim,
            num_heads=self.num_heads,
            zip_frame_buckets=[1, 2, 16],
            drop_mode=framepack_drop_mode) # padd

        self.block_mask = None

        self.gradient_checkpointing = False

        self.pre_compute_freqs = {}
        self.freq_prev_frame_seqlens = {}
        self.freqs_t_cache = {}
        self.freqs_t_cache['freqs_t_prefill'] = get_rope_t_params(t=30, num_frame_per_block=1, d=self.dim // self.num_heads, device='cuda')

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)

            # Audio injector zero init will be implemented when audio components are added
            # for i in range(self.audio_injector.injector.__len__()):
            #     self.audio_injector.injector[i].o = zero_module(
            #         self.audio_injector.injector[i].o)

    def process_motion_frame_pack(self,
                                  motion_latents,
                                  drop_motion_frames=False,
                                  add_last_motion=2,
                                  sp_dim=None,
                                  rope_recompute_needed=True):
        flat_mot, mot_rope_emb = self.frame_packer(motion_latents,
                                                   add_last_motion,
                                                   sp_dim=sp_dim,
                                                   rope_recompute_needed=rope_recompute_needed)
        if motion_latents is None:
            return [m[:, :0] for m in flat_mot], [m[:, :0] for m in mot_rope_emb]
        else:
            return flat_mot, mot_rope_emb

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True,
                      sp_dim=None,
                      rope_recompute_needed=True):
        # inject the motion frames token to the hidden states
        if isinstance(motion_latents, list):
            assert len(motion_latents) == 1
        elif isinstance(motion_latents, torch.Tensor):
            assert motion_latents.shape[0] == 1
        motion_latents, motion_rope_emb = self.process_motion_frame_pack(
                        motion_latents,
                        drop_motion_frames=drop_motion_frames,
                        add_last_motion=add_last_motion,
                        sp_dim=sp_dim,
                        rope_recompute_needed=rope_recompute_needed)

        #if len(motion_latents) > 0:
        if motion_latents is not None:
            x = torch.cat([motion_latents, x], dim=1)
            #x = [torch.cat([m, u], dim=1) for m, u in zip(motion_latents, x)]
            seq_lens += motion_latents.shape[1]
            #seq_lens = torch.tensor([r.size(1) for r in motion_latents], dtype=torch.long) + seq_lens

            #rope_embs = [
            #    torch.cat([m, u], dim=1) for m, u in zip(motion_rope_emb, rope_embs)
            #]
            if rope_recompute_needed:
                rope_embs = torch.cat([motion_rope_emb, rope_embs], dim=1)
            else:
                rope_embs = None
            mask_input = torch.cat([torch.full([mask_input.shape[0], motion_latents.shape[1]], fill_value=2, device=mask_input.device, dtype=mask_input.dtype), mask_input], dim=1)
                #mask_input = [torch.cat([2 * torch.ones([1, u.shape[1]], device=m.device, dtype=m.dtype), m], dim=1) for u, m in zip(motion_latents, mask_input)]

        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
        # TODO: optimize time cost
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_frames = audio_emb.shape[1]

            input_hidden_states = hidden_states[:, -self.
                                                original_seq_len:].clone(
                                                )  # b (f h w) c
            input_hidden_states = rearrange(
                input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enbale_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,
                                             "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id](
                        input_hidden_states, temb=audio_emb_global[:, 0])
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                    audio_attn_id](
                        input_hidden_states)
            audio_emb = rearrange(
                audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(
                    attn_hidden_states.shape[0],
                    dtype=torch.long,
                    device=attn_hidden_states.device) * attn_audio_emb.shape[1])
            residual_out = rearrange(
                residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            # hidden_states[:, -self.
            #               original_seq_len:] = hidden_states[:, -self.
            #                                                 original_seq_len:] + residual_out
            condition_hidden_states = hidden_states[:, :-self.original_seq_len]
            video_hidden_states = hidden_states[:, -self.original_seq_len:]
            video_hidden_states = video_hidden_states + residual_out
            hidden_states = torch.cat([condition_hidden_states, video_hidden_states], dim=1)

        return hidden_states

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_s2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1, is_sparse=False,
        motion_latents_seqlen=0
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [N latent frame] ... [N latent frame](video) [1 latent frame]
        The first frame is separated out to support S2V generation
        We use flexattention to construct the attention mask
        """
        condition_length = motion_latents_seqlen + frame_seqlen
        total_length = num_frames * frame_seqlen + condition_length
        block_length = frame_seqlen * num_frame_per_block

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        ends[:condition_length] = condition_length

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=condition_length,
            end=total_length,
            step=block_length,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + block_length] = tmp + block_length

        def attention_mask_causal(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        def attention_mask_sparse_casual(b, h, q_idx, kv_idx): # local_attn_size ignored
            return (kv_idx < condition_length) | ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * block_length))) | (q_idx == kv_idx)

        attention_mask = attention_mask_causal if not is_sparse else attention_mask_sparse_casual

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
            self,
            x,
            t,
            context,
            seq_len,
            ref_latents,
            motion_latents=None,
            audio_input=None,
            motion_frames=[73, 19],
            add_last_motion=2,
            drop_motion_frames=False,
            kv_cache: dict = None,
            crossattn_cache: dict = None,
            current_start: int = 0,
            cache_start: int = 0,
            sp_dim=None,
            initial_ref=False,
            slice_index=None,
            sink_size=None,
            **kwargs,
    ):
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B, T].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      [B, C, T_m, H, W].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
        drop_motion_frames  Bool, whether drop the motion frames info
        sp_dim              The dimension of the sequence parallel
        initial_ref         Bool, whether this is the first forward, only includes ref image and motion latents
        """
        device = self.patch_embedding.weight.device
        is_equal = torch.all(t == t[:, :1], dim=1)
        assert torch.all(is_equal), f"{t}"
        t = t[:, 0]
        if not self.freqs.device == device:
            self.freqs = self.freqs.to(device)

        if initial_ref:
            if isinstance(ref_latents, torch.Tensor):
                B, C, T, H, W = ref_latents.shape
                T = self.max_latent_frames
            else: # list
                B = len(ref_latents)
                C, T, H, W = ref_latents[0].shape
                T = self.max_latent_frames

            frame_seqlen = H * W // self.patch_size[-2] // self.patch_size[-1]
            add_last_motion = int(self.add_last_motion) * add_last_motion
            grid_sizes = torch.tensor([[T, H // 2, W // 2]]).repeat(B, 1)
            seq_lens = [T * H * W // 4] * B
            self.original_seq_len = seq_lens[0]
            grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

            # ref and motion
            ref = self.patch_embedding(ref_latents) # replacement?
            batch_size = ref.shape[0]
            height, width = ref.shape[-2:]
            ref_grid_sizes = [[
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),  # the start index
                torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),  # the end index
                torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),]]  # the range

            ref = ref.flatten(2).transpose(1, 2) # b, s, c
            seq_lens = ref.shape[1]
            mask_input = torch.ones([B, ref.shape[1]], dtype=torch.long, device=device)
            grid_sizes = ref_grid_sizes + grid_sizes

            x = ref

            b = batch_size
            s = frame_seqlen * (T + 1)
            n = self.num_heads
            d = self.dim // self.num_heads

            if self.freq_prev_frame_seqlens.get('no_motion_latents', None) != frame_seqlen or self.pre_compute_freqs.get('no_motion_latents', None) is None:
                pre_compute_freqs = rope_precompute([b, s, n, d], grid_sizes, self.freqs, start=None, sp_dim=sp_dim)
                self.freq_prev_frame_seqlens['no_motion_latents'] = frame_seqlen
                self.pre_compute_freqs['no_motion_latents'] = pre_compute_freqs
            else:
                pre_compute_freqs = self.pre_compute_freqs['no_motion_latents']

            if motion_latents is not None:
                rope_recompute_needed = self.freq_prev_frame_seqlens.get('with_motion_latents', None) != frame_seqlen or self.pre_compute_freqs.get('with_motion_latents', None) is None
                x, seq_lens, pre_compute_freqs, mask_input = self.inject_motion(
                    x,
                    seq_lens,
                    pre_compute_freqs,
                    mask_input,
                    motion_latents,
                    drop_motion_frames=drop_motion_frames,
                    add_last_motion=add_last_motion,
                    sp_dim=sp_dim,
                    rope_recompute_needed=rope_recompute_needed)
                if rope_recompute_needed:
                    self.freq_prev_frame_seqlens['with_motion_latents'] = frame_seqlen
                    self.pre_compute_freqs['with_motion_latents'] = pre_compute_freqs
                else:
                    pre_compute_freqs = self.pre_compute_freqs['with_motion_latents']

            pre_compute_freqs = pre_compute_freqs[0]
            if NO_REFRESH_INFERENCE:
                self.freqs_t_cache['freqs_i_prefill'] = precompute_freqs_i(convert_grid_sizes(grid_sizes, current_start), self.freqs_t_cache['freqs_t_prefill'], self.freqs, self.dim // self.num_heads, start_frame=0, sp_dim=sp_dim)

        else:
            B, C, T, H, W = x.shape
            frame_seqlen = H * W // self.patch_size[-2] // self.patch_size[-1]
            audio_input = torch.cat([
                audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input
            ], dim=-1)
            audio_emb_res = self.casual_audio_encoder(audio_input)
            if self.enbale_adain:
                audio_emb_global, audio_emb = audio_emb_res # [1, 39, 1, 5120], [1, 39, 5, 5120]
                self.audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
                self.audio_emb_global = self.audio_emb_global[:, slice_index[0]:slice_index[1]].to(torch.bfloat16)
            else:
                audio_emb = audio_emb_res
            self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
            self.merged_audio_emb = self.merged_audio_emb[:, slice_index[0]:slice_index[1]].to(torch.bfloat16)

            # embeddings
            x = self.patch_embedding(x)
            grid_sizes = torch.tensor([[T, H // 2, W // 2]]).repeat(B, 1)

            x = x.flatten(2).transpose(1, 2)
            seq_lens = x.size(1)
            mask_input = torch.zeros([B, x.shape[1]], dtype=torch.long, device=x[0].device)
            self.original_seq_len = x.size(1)
            pre_compute_freqs = self.pre_compute_freqs['no_motion_latents'] if motion_latents is None else self.pre_compute_freqs['with_motion_latents']
            pre_compute_freqs = pre_compute_freqs[0]

        current_frame = current_start // frame_seqlen
        if NO_REFRESH_INFERENCE and current_frame != self.freqs_t_cache.get('frame_no', None):
            self.freqs_t_cache['frame_no'] = current_frame
            freqs_t = get_rope_t_params(t=current_frame, num_frame_per_block=self.num_frame_per_block, d=self.dim // self.num_heads, device=self.device)
            self.freqs_t_cache['freqs_i'] = precompute_freqs_i(convert_grid_sizes(grid_sizes, current_start), freqs_t, self.freqs, self.dim // self.num_heads, start_frame=0, sp_dim=sp_dim)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(torch.bfloat16)) # B, C
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)) # B, 6, C

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes if not NO_REFRESH_INFERENCE else convert_grid_sizes(grid_sizes, current_start),
            freqs=self.freqs if NO_REFRESH_INFERENCE else pre_compute_freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            frame_seqlen=frame_seqlen,
            sp_dim=sp_dim,
            sink_size=sink_size,
            freqs_i = (self.freqs_t_cache['freqs_i_prefill'] if initial_ref else self.freqs_t_cache['freqs_i']) if NO_REFRESH_INFERENCE else None,
            mode='dmd')

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                if not initial_ref:
                    # x = torch.utils.checkpoint.checkpoint(
                    #     create_custom_forward(self.after_transformer_block),
                    #     block_index, x,
                    #     use_reentrant=False,
                    # )
                    x = self.after_transformer_block(block_index, x)
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)
                if not initial_ref:
                    x = self.after_transformer_block(block_index, x)

        if initial_ref:
            return x.shape[1]

        # head
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
            self,
            x,
            t,
            context,
            seq_len,
            ref_latents,
            motion_latents=None,
            audio_input=None,
            motion_frames=[73, 19],
            add_last_motion=2,
            drop_motion_frames=False,
            sp_dim=None,
            sink_size=None,
            **kwargs,
    ):
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B, T].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      [B, C, T_m, H, W].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
        drop_motion_frames  Bool, whether drop the motion frames info
        sp_dim              The dimension of the sequence parallel
        """
        B, C, T, H, W = x.shape
        frame_seqlen = H * W // self.patch_size[-2] // self.patch_size[-1]

        add_last_motion = self.add_last_motion * add_last_motion

        audio_input = torch.cat([
            audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input
        ],
                                dim=-1)
        audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res # [1, 39, 1, 5120], [1, 39, 5, 5120]
            self.audio_emb_global = audio_emb_global[:,
                                                     motion_frames[1]:].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1]:, :]

        device = self.patch_embedding.weight.device

        # Construct blockwise causal attn mask
        current_frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]) * mpu.get_sequence_parallel_world_size()
        motion_latents_seqlen = current_frame_seqlen * 3 // 2 if motion_latents is not None else 0
        num_frames = x.shape[2]
        if self.frame_seqlen_cache is None or self.frame_seqlen_cache != current_frame_seqlen or self.motion_latents_seqlen_cache != motion_latents_seqlen:
            self.frame_seqlen_cache = current_frame_seqlen
            self.motion_latents_seqlen_cache = motion_latents_seqlen
            self.block_mask = self._prepare_blockwise_causal_attn_mask_s2v(
                device, num_frames=num_frames,
                frame_seqlen=current_frame_seqlen,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size,
                is_sparse=self.is_sparse,
                motion_latents_seqlen=motion_latents_seqlen
            )

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # ref and motion
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [[
            torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size,
                                                         1),  # the start index
            torch.tensor([31, height,
                          width]).unsqueeze(0).repeat(batch_size,
                                                      1),  # the end index
            torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
        ]  # the range
                         ]

        ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
        self.original_seq_len = seq_lens[0]

        seq_lens = torch.tensor([r.size(1) for r in ref], dtype=torch.long) + seq_lens

        grid_sizes = ref_grid_sizes + grid_sizes
        x = [torch.cat([r, u], dim=1) for r, u in zip(ref, x)]

        # Initialize masks to indicate noisy latent, ref latent, and motion latent.
        # However, at this point, only the first two (noisy and ref latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
        mask_input = [
            torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device)
            for u in x
        ]
        for i in range(len(mask_input)):
            mask_input[i][:, :-self.original_seq_len] = 1

        # compute the rope embeddings for the input
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(
            1), self.num_heads, self.dim // self.num_heads
        #self.pre_compute_freqs = rope_precompute(
        #    x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None, sp_dim=sp_dim)
        self.pre_compute_freqs = rope_precompute(
            [b, s, n, d], grid_sizes, self.freqs, start=None, sp_dim=sp_dim)

        #x = [u.unsqueeze(0) for u in x]
        #self.pre_compute_freqs = [
        #    u.unsqueeze(0) for u in self.pre_compute_freqs
        #]

        if motion_latents is not None:
            x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
                x,
                seq_lens,
                self.pre_compute_freqs,
                mask_input,
                motion_latents,
                drop_motion_frames=drop_motion_frames,
                add_last_motion=add_last_motion,
                sp_dim=sp_dim)

        #x = torch.cat(x, dim=0)
        #self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # time embeddings
        if self.zero_timestep: # default: True
            t = torch.cat([t, torch.zeros([t.shape[0], 1], dtype=t.dtype, device=t.device)], dim=1) # B, T+1
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).float().unflatten(dim=0, sizes=t.shape)) # B, T+1, C
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, T+1, 6, C
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if self.zero_timestep:
            e = e[:, :-1]
            token_len = x.shape[1]
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        self.pre_compute_freqs = self.pre_compute_freqs[0]
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            frame_seqlen=frame_seqlen,
            sp_dim=sp_dim,
            mode='ode')

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.after_transformer_block),
                    idx, x,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
                x = self.after_transformer_block(idx, x)

        # unpatchify
        x = x[:, -self.original_seq_len:]
        # head
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)