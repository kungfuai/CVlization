## Quickstart

```
bash examples/image_gen/dit/build.sh
bash examples/image_gen/dit/build_data.sh
bash examples/image_gen/dit/train.sh
```

This is tested on GTX 3090. It uses flash attention, which requires sm80 or newer gen GPUs.

## Model architecture

This is the DIT-XL/2 architecture:

```
DiT(
  (x_embedder): PatchEmbed(
    (proj): Conv2d(4, 1152, kernel_size=(2, 2), stride=(2, 2))
    (norm): Identity()
  )
  (t_embedder): TimestepEmbedder(
    (mlp): Sequential(
      (0): Linear(in_features=256, out_features=1152, bias=True)
      (1): SiLU()
      (2): Linear(in_features=1152, out_features=1152, bias=True)
    )
  )
  (y_embedder): LabelEmbedder(
    (embedding_table): Embedding(1001, 1152)
  )
  (blocks): ModuleList(
    (0-27): 28 x DiTBlock(
      (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
      (attn): Attention(
        (qkv): Linear(in_features=1152, out_features=3456, bias=True)
        (q_norm): Identity()
        (k_norm): Identity()
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=1152, out_features=1152, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
      (mlp): Mlp(
        (fc1): Linear(in_features=1152, out_features=4608, bias=True)
        (act): GELU(approximate='tanh')
        (drop1): Dropout(p=0, inplace=False)
        (norm): Identity()
        (fc2): Linear(in_features=4608, out_features=1152, bias=True)
        (drop2): Dropout(p=0, inplace=False)
      )
      (adaLN_modulation): Sequential(
        (0): SiLU()
        (1): Linear(in_features=1152, out_features=6912, bias=True)
      )
    )
  )
  (final_layer): FinalLayer(
    (norm_final): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
    (linear): Linear(in_features=1152, out_features=32, bias=True)
    (adaLN_modulation): Sequential(
      (0): SiLU()
      (1): Linear(in_features=1152, out_features=2304, bias=True)
    )
  )
)
```

## Reference

- Adapted from https://github.com/chuanyangjin/fast-DiT.
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)