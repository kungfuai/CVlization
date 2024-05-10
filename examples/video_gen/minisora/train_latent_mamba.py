import numpy as np
from cvlization.torch.training_pipeline.lm.mamba import MambaTrainingPipeline
from cvlization.torch.training_pipeline.lm.data_utils import FlatTokenIds


def prepare_data(args):
    data = np.load(args.tokens_input_file).astype(np.uint16)
    print(data.shape)
    data = data[:, :2, :, :]
    vae_vocab_size = data.max() + 1
    vocab_size = data.max() + 3
    VIDEO_BEGIN_TOKEN = data.max() + 1
    position_shape = data.shape[1:]
    data = data.reshape(len(data), -1)  # flattened for each video
    
    return data, position_shape, vae_vocab_size, vocab_size, VIDEO_BEGIN_TOKEN


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/latent-mamba")
    parser.add_argument("--project", type=str, default="flying_mnist")
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_embed", type=int, default=128 * 6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--vae_model_name", type=str, default="zzsi_kungfu/videogpt/model-kbu39ped:v11"
    )
    parser.add_argument(
        "--tokens_input_file",
        type=str,
        default="flying_mnist_tokens_32frames_train.npy",
    )
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    token_ids, position_shape, vae_vocab_size, vocab_size, VIDEO_BEGIN_TOKEN = (
        prepare_data(args)
    )
    latent_sequence_length = int(np.prod(token_ids.shape[1:]))
    print(f"latent_sequence_length: {latent_sequence_length}")
    dataset_builder = FlatTokenIds(
        token_ids=token_ids, vocab_size=vocab_size, start_token_id=VIDEO_BEGIN_TOKEN
    )
    train_pipe = MambaTrainingPipeline(
        config=MambaTrainingPipeline.Config(
            output_dir=args.log_dir,
            vae_vocab_size=vae_vocab_size,
            vocab_size=vocab_size,
            start_token=VIDEO_BEGIN_TOKEN,
            project=args.project,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
            sample_interval=args.sample_interval,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            position_shape=position_shape,
            block_size=args.block_size,
            d_model=args.n_embed,
            n_layer=args.n_layer,
            lr=args.learning_rate,
            max_iters=args.max_iters,
            max_length_to_generate=latent_sequence_length,
            # lr_decay_iters=args.lr_decay_iters,
            # min_lr=args.min_lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            vae_model_name=args.vae_model_name,
            track=args.track,
        )
    )
    train_pipe.fit(dataset_builder)


if __name__ == "__main__":
    main()
