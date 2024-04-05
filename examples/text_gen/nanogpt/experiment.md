```
bash examples/text_gen/nanogpt/train.sh --wandb_log=True --block_size 512
bash examples/text_gen/nanogpt/train.sh --wandb_log=True --sparse_context_window=True --block_size=512 --context_stride=2 --context_stride_start=32
bash examples/text_gen/nanogpt/train.sh --wandb_log=True --sparse_context_window=True --block_size=512 --context_stride=3 --context_stride_start=32
bash examples/text_gen/nanogpt/train.sh --wandb_log=True --sparse_context_window=True --block_size=512 --context_stride=3 --context_stride_start=64
bash examples/text_gen/nanogpt/train.sh --wandb_log=True --sparse_context_window=True --block_size=512 --context_stride=4 --context_stride_start=64
```