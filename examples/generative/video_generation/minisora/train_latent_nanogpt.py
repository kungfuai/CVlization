import numpy as np
from cvlization.torch.training_pipeline.lm.gpt import NanoGPTTrainingPipeline
from cvlization.torch.training_pipeline.lm.data_utils import FlatTokenIds


def prepare_data(args):
    print("Loading data from", args.tokens_input_file)
    data = np.load(args.tokens_input_file).astype(np.uint16)
    vae_vocab_size = data.max() + 1
    vocab_size = data.max() + 2
    VIDEO_BEGIN_TOKEN = data.max() + 1
    data = data.reshape(len(data), -1)  # flattened for each video
    return data, vae_vocab_size, vocab_size, VIDEO_BEGIN_TOKEN


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/latent-nanogpt")
    parser.add_argument("--project", type=str, default="flying_mnist")
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=128 * 6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--vae_model_name", type=str, default="zzsi_kungfu/videogpt/model-kbu39ped:v11")
    parser.add_argument("--tokens_input_file", type=str, default="data/latents/flying_mnist_11k__model-kbu39ped_tokens_32frames_train.npy")
    parser.add_argument("--sparse_context_window", action="store_true")
    parser.add_argument("--context_stride", type=int, default=2)
    parser.add_argument("--context_stride_start", type=int, default=32)
    parser.add_argument("--mamba", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    token_ids, vae_vocab_size, vocab_size, VIDEO_BEGIN_TOKEN = prepare_data(args)
    print(token_ids.shape)
    dataset_builder = FlatTokenIds(token_ids=token_ids, vocab_size=vocab_size, start_token_id=VIDEO_BEGIN_TOKEN)
    train_pipe = NanoGPTTrainingPipeline(
        config=NanoGPTTrainingPipeline.Config(
            log_dir=args.log_dir,
            project=args.project,
            vae_model_name=args.vae_model_name,
            vae_vocab_size=vae_vocab_size,
            vocab_size=vocab_size,
            start_token=VIDEO_BEGIN_TOKEN,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
            sample_interval=args.sample_interval,
            batch_size=args.batch_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            max_iters=args.max_iters,
            lr_decay_iters=args.lr_decay_iters,
            min_lr=args.min_lr,
            beta2=args.beta2,
            warmup_iters=args.warmup_iters,
            wandb_log=args.wandb_log,
            sparse_context_window=args.sparse_context_window,
            context_stride=args.context_stride,
            context_stride_start=args.context_stride_start,
            use_mamba_mixer=args.mamba,
            compile=args.compile,
        )
    )
    train_pipe.fit(dataset_builder)
    

if __name__ == "__main__":
    main()