# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from importlib.metadata import version
from mmgp import offload
import torch.nn.functional as F


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn = None

try:
    from sageattention import sageattn_varlen
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ):
        return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
except ImportError:
    sageattn_varlen_wrapper = None


import warnings

try:
    from sageattention import sageattn
    @torch.compiler.disable()
    def sageattn_wrapper(
            qkv_list,
            attention_length
        ):
        q,k, v = qkv_list
        padding_length = q.shape[0] -attention_length
        q = q[:attention_length, :, : ].unsqueeze(0)
        k = k[:attention_length, :, : ].unsqueeze(0)
        v = v[:attention_length, :, : ].unsqueeze(0)

        o = sageattn(q, k, v, tensor_layout="NHD").squeeze(0)
        del q, k ,v
        qkv_list.clear()

        if padding_length > 0:
            o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

        return o
except ImportError:
    sageattn = None


@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    padding_length = q.shape[0] -attention_length
    q = q[:attention_length, :].transpose(0,1).unsqueeze(0)
    k = k[:attention_length, :].transpose(0,1).unsqueeze(0)
    v = v[:attention_length, :].transpose(0,1).unsqueeze(0)

    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False
    ).squeeze(0).transpose(0,1)
    del q, k ,v
    qkv_list.clear()

    if padding_length > 0:
        o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    return o


def get_attention_modes():
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    # if memory_efficient_attention != None:
    #     ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn != None and version("sageattention").startswith("2") :
        ret.append("sage2")

    return ret


__all__ = [
    'pay_attention',
    'attention',
]


def pay_attention(
    qkv_list,
    # q,
    # k,
    # v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    force_attention= None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """

    attn = offload.shared_state["_attention"] if force_attention== None else force_attention
    q,k,v = qkv_list
    qkv_list.clear()

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if attn=="sage":
        x = sageattn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_kv=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
        ).unflatten(0, (b, lq))
    elif attn=="sage2":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn_wrapper(qkv_list, lq).unsqueeze(0)
    elif attn=="sdpa":
        qkv_list = [q, k, v]
        del q, k , v
        x = sdpa_wrapper( qkv_list, lq).unsqueeze(0)
    elif attn=="flash" and (version is None or version == 3):
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif attn=="flash":
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return pay_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
