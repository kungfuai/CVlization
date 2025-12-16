# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat
from torch.distributed.tensor import distribute_tensor
from .inference_utils import conditional_compile

from .model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanModel,
    WanSelfAttention,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
)

from .audio_utils import CausalAudioEncoder, AudioInjector_WAN

from ...utils import parallel_state as mpu
from ...utils.all_to_all import SeqAllToAll4D


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


@amp.autocast(enabled=False)
@conditional_compile
def rope_apply(x, grid_sizes, freqs, sp_dim=None, current_start=0):
    n, c = x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[current_start:current_start + s]
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()



def rope_precompute(shape, grid_sizes, freqs, start=None, sp_dim=None):
    #b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    b, s, n, c = shape
    c = c // 2
    sp_size = mpu.get_sequence_parallel_world_size()
    sp_rank = mpu.get_sequence_parallel_rank()

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    #output = torch.view_as_complex(x.detach().reshape(b, s, n, -1,
    #                                                  2).to(torch.float64))
    output = torch.empty([b, s, n, c], dtype=torch.complex128, device='cuda')
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        #batch_size = len(g[0])
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                # actually not used
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o

            # add sequence parallel rank to the start index
            if sp_dim == 'h' and sp_size > 1:
                h_o = h_o + sp_rank * t_h
            elif sp_dim == 'w' and sp_size > 1:
                w_o = w_o + sp_rank * t_w
            else:
                assert sp_size == 1, 'sp_size > 1 but sp_dim not specified'

            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (t_h / seq_h).item(), (t_w / seq_w).item()
                    #factor_f, factor_h, factor_w = (t_f / seq_f), (t_h / seq_h), (t_w / seq_w)
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())

                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1,
                                            seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(),
                                            (-t_f - f_o).item() + 1,
                                            seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1,
                                        seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1,
                                        seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][
                        f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat([
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                    ], dim=-1).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output

class WanS2VSelfAttention(WanSelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs, sp_dim=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q=rope_apply(q, grid_sizes, freqs, sp_dim=sp_dim)
        k=rope_apply(k, grid_sizes, freqs, sp_dim=sp_dim)

        if self.sp_size > 1:
            gqa_backward_allreduce = False
            q = SeqAllToAll4D.apply(self.sp_group, q, self.scatter_idx, self.gather_idx, False)
            k = SeqAllToAll4D.apply(self.sp_group, k, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)
            v = SeqAllToAll4D.apply(self.sp_group, v, self.scatter_idx, self.gather_idx, gqa_backward_allreduce)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens * mpu.get_sequence_parallel_world_size(),
            window_size=self.window_size)

        if self.sp_size > 1:
            x = SeqAllToAll4D.apply(self.sp_group, x, self.gather_idx, self.scatter_idx, False)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanS2VAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__('t2v_cross_attn', dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps)
        self.self_attn = WanS2VSelfAttention(dim, num_heads, window_size,
                                             qk_norm, eps)


    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        # assert e[0].dtype == torch.float32
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.size(1))
        seg_idx = [0, x.size(1) - seg_idx, x.size(1)]
        e = e[0] # B,6,2,C
        modulation = self.modulation.unsqueeze(2)
        # with amp.autocast(dtype=torch.float32):
        e = (modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        e = [element.squeeze(1) for element in e]
        norm_x = self.norm1(x).float()
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i]:seg_idx[i + 1]] *
                         (1 + e[1][:, i:i + 1]) + e[0][:, i:i + 1])
        norm_x = torch.cat(parts, dim=1)
        # self-attention
        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        # with amp.autocast(dtype=torch.float32):
        z = []
        for i in range(2):
            z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[2][:, i:i + 1])
        y = torch.cat(z, dim=1)
        x = x + y
        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            norm2_x = self.norm2(x).float()
            parts = []
            for i in range(2):
                parts.append(norm2_x[:, seg_idx[i]:seg_idx[i + 1]] *
                             (1 + e[4][:, i:i + 1]) + e[3][:, i:i + 1])
            norm2_x = torch.cat(parts, dim=1)
            y = self.ffn(norm2_x)
            # with amp.autocast(dtype=torch.float32):
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i]:seg_idx[i + 1]] * e[5][:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class Head_S2V(Head):

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x

class WanModel_S2V(ModelMixin, ConfigMixin):
    ignore_for_config = [
        'args', 'kwargs', 'patch_size', 'cross_attn_norm', 'qk_norm',
        'text_dim', 'window_size'
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
            enable_tsm=False,
            trainable_token_pos_emb=False,
            motion_token_num=1024,
            enable_framepack=True,  # Mutually exclusive with enable_motioner
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
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            *args,
            **kwargs):
        super().__init__()

        assert model_type == 's2v'
        self.model_type = model_type

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
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

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
            WanS2VAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                                 cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head_S2V(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

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
        self.trainable_token_pos_emb = trainable_token_pos_emb

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode) # padd

        self.gradient_checkpointing = False

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
                                  sp_dim=None):
        flattern_mot, mot_remb = self.frame_packer(motion_latents,
                                                   add_last_motion,
                                                   sp_dim=sp_dim)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot
                   ], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def inject_motion(self,
                      x,
                      seq_lens,
                      rope_embs,
                      mask_input,
                      motion_latents,
                      drop_motion_frames=False,
                      add_last_motion=True,
                      sp_dim=None):
        # inject the motion frames token to the hidden states
        motion_latents, motion_rope_emb = self.process_motion_frame_pack(
                        motion_latents,
                        drop_motion_frames=drop_motion_frames,
                        add_last_motion=add_last_motion,
                        sp_dim=sp_dim)

        if len(motion_latents) > 0:
            x = [torch.cat([m, u], dim=1) for m, u in zip(motion_latents, x)]
            seq_lens = torch.tensor([r.size(1) for r in motion_latents], dtype=torch.long) + seq_lens
            rope_embs = [
                torch.cat([m, u], dim=1) for m, u in zip(motion_rope_emb, rope_embs)
            ]
            mask_input = [
                torch.cat([
                    2 * torch.ones([1, u.shape[1] - m.shape[1]],
                                      device=m.device,
                                      dtype=m.dtype),
                    m
                ],
                          dim=1) for u, m in zip(x, mask_input)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
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
                # FIXME: sometimes self.audio_injector will have parameters type DTensor, but audio_emb_global is not DTensor
                # is_audio_injector_DTensor = any(isinstance(p, torch.distributed.tensor.DTensor) for p in self.audio_injector.parameters())
                # if is_audio_injector_DTensor:
                #     mesh = next(p._spec.mesh for p in self.audio_injector.parameters() if hasattr(p, '_spec'))
                #     audio_injector_temb = distribute_tensor(audio_emb_global[:, 0], mesh)
                #     input_hidden_states = distribute_tensor(input_hidden_states, mesh)
                # else:
                #     audio_injector_temb = audio_emb_global[:, 0]
                audio_injector_temb = audio_emb_global[:, 0]
                adain_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id](
                        input_hidden_states, temb=audio_injector_temb)
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

    def audio_injection(self, block_idx, hidden_states):
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
                audio_injector_temb = audio_emb_global[:, 0]
                adain_hidden_states = self.audio_injector.injector_adain_layers[
                    audio_attn_id](
                        input_hidden_states, temb=audio_injector_temb)
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[
                    audio_attn_id](
                        input_hidden_states)
            audio_emb = rearrange(
                audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb

            def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    return module(*inputs, **kwargs)
                return custom_forward

            context_lens = torch.ones(attn_hidden_states.shape[0], dtype=torch.long, device=attn_hidden_states.device)*attn_audio_emb.shape[1]
            residual_out = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.audio_injector.injector[audio_attn_id]),
                attn_hidden_states, attn_audio_emb, context_lens,
                use_reentrant=False
            )
            residual_out = rearrange(
                residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            condition_hidden_states = hidden_states[:, :-self.original_seq_len]
            video_hidden_states = hidden_states[:, -self.original_seq_len:]
            video_hidden_states = video_hidden_states + residual_out
            hidden_states = torch.cat([condition_hidden_states, video_hidden_states], dim=1)

        return hidden_states

    def forward(
            self,
            x,
            t,
            context,
            seq_len,
            ref_latents,
            motion_latents=None,
            audio_input=None,
            classify_mode=False,
            concat_time_embeddings=False,
            register_tokens=None,
            cls_pred_branch=None,
            gan_ca_blocks=None,
            y=None,
            clip_feature=None,
            motion_frames=[73, 19],
            add_last_motion=2,
            drop_motion_frames=False,
            sp_dim=None,
            gan_mode="generator",
            **kwargs
    ):
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      [B, C, T_m, H, W].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
        drop_motion_frames  Bool, whether drop the motion frames info
        sp_dim              The dimension of the sequence parallel
        """
        forward_env = torch.no_grad() if gan_mode == "discriminator" else nullcontext()
        with forward_env:
            print(f'sp_dim: {sp_dim}')
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

            x = [u.unsqueeze(0) for u in x]
            self.pre_compute_freqs = [
                u.unsqueeze(0) for u in self.pre_compute_freqs
            ]

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

            x = torch.cat(x, dim=0)
            self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
            mask_input = torch.cat(mask_input, dim=0)

            x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

            # time embeddings
            if self.zero_timestep: # default: True
                t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
            # with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                # assert e.dtype == torch.float32 and e0.dtype == torch.float32

            if self.zero_timestep:
                e = e[:-1]
                zero_e0 = e0[-1:]
                e0 = e0[:-1]
                token_len = x.shape[1]
                e0 = torch.cat([
                    zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1),
                    e0.unsqueeze(2)
                ],
                            dim=2)
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
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # For DMD2Custom training
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(None), "n d -> b n d", b=x.shape[0])

        gan_idx = 0
        for idx, block in enumerate(self.blocks):
            with forward_env:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, **kwargs,
                        use_reentrant=False,
                    )

                    x = self.audio_injection(idx, x)
                else:
                    x = block(x, **kwargs)
                    x = self.after_transformer_block(idx, x)

            if classify_mode and idx in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                if gan_mode == "discriminator":
                    # add detach() to x to avoid the discriminator update influence fake_score
                    final_x.append(gan_ca_blocks[gan_idx](x.detach(), gan_token))
                else:
                    final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :].detach()], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        with forward_env:
            # unpatchify
            x = x[:, -self.original_seq_len:]
            # head
            x = self.head(x, e)
            x = self.unpatchify(x, original_grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x
        return torch.stack(x)

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


class FramePackMotioner(nn.Module):

    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[
                1, 2, 16
            ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(
            16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(
            16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(
            16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(
            zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim %
                num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        d = inner_dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2, sp_dim=None, rope_recompute_needed=True):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        #for m in motion_latents:
        m = motion_latents[0]

        lat_height, lat_width = m.shape[2], m.shape[3]
        padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height,
                                lat_width).to(
                                    device=m.device, dtype=m.dtype)
        overlap_frame = min(padd_lat.shape[1], m.shape[1])
        if overlap_frame > 0:
            padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.
                                                    __len__() -
                                                    add_last_motion -
                                                    1].sum()
            padd_lat[:, -zero_end_frame:] = 0

        padd_lat = padd_lat.unsqueeze(0)
        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum(
        ):, :, :].split(
            list(self.zip_frame_buckets)[::-1], dim=2)  # 16, 2 ,1

        # patchfy
        clean_latents_post = self.proj(clean_latents_post).flatten(
            2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(
            2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(
            2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :
                                                    0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :
                                                0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat(
            [clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        # rope
        start_time_id = -(self.zip_frame_buckets[:1].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[0]
        grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                    [
                        [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                    ]
        #grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else [[[[start_time_id, 0, 0]], [[end_time_id, lat_height // 2, lat_width // 2]], [[self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]]]]

        start_time_id = -(self.zip_frame_buckets[:2].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
        grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
        [
            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
            torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
            torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
        ]
        #grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else [[[[start_time_id, 0, 0]], [[end_time_id, lat_height // 4, lat_width // 4]], [[self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]]]]

        start_time_id = -(self.zip_frame_buckets[:3].sum())
        end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
        grid_sizes_4x = [[
            torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
            torch.tensor([end_time_id, lat_height // 8,
                        lat_width // 8]).unsqueeze(0).repeat(1, 1),
            torch.tensor([
                self.zip_frame_buckets[2], lat_height // 2, lat_width // 2
            ]).unsqueeze(0).repeat(1, 1),
        ]]
        #grid_sizes_4x = [[[[start_time_id, 0, 0]], [[end_time_id, lat_height // 8, lat_width // 8]], [[self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]]]]

        grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

        '''
        motion_rope_emb = rope_precompute(
            motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads,
                                        self.inner_dim // self.num_heads),
            grid_sizes,
            self.freqs,
            start=None,
            sp_dim=sp_dim)
        '''
        if rope_recompute_needed:
            motion_rope_emb = rope_precompute(
                [1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads],
                grid_sizes,
                self.freqs,
                start=None,
                sp_dim=sp_dim)
        else:
            motion_rope_emb = None

        #    mot.append(motion_lat)
        #    mot_remb.append(motion_rope_emb)
        #return mot, mot_remb
        return motion_lat, motion_rope_emb