import numpy as np
from cvlization.torch.training_pipeline.lm.mdgpt import MDGPTTrainingPipeline


def prepare_data(args):
    data = np.load(args.tokens_input_file).astype(np.uint16)
    vocab_size = data.max() + 3
    VIDEO_BEGIN_TOKEN = data.max() + 1
    IGNORE_TOKEN = data.max() + 2
    # data = data.reshape(len(data), -1)  # flattened for each video
    return data, vocab_size, VIDEO_BEGIN_TOKEN, IGNORE_TOKEN

class SimplePassthroughDatasetBuilder:
    def __init__(self, token_ids, vocab_size, VIDEO_BEGIN_TOKEN):
        self.token_ids = token_ids
        self.vocab_size = vocab_size
        self.start_token = VIDEO_BEGIN_TOKEN
        self.n_train = int(len(token_ids) * 0.8)

    def training_dataset(self):
        return self.token_ids[: self.n_train]

    def validation_dataset(self):
        return self.token_ids[self.n_train:]
    


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
    parser.add_argument("--sparse_block_size", type=int, default=128)
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
    parser.add_argument("--tokens_input_file", type=str, default="flying_mnist_tokens_32frames_train.npy")
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    token_ids, vocab_size, VIDEO_BEGIN_TOKEN, IGNORE_TOKEN = prepare_data(args)
    db = SimplePassthroughDatasetBuilder(token_ids, vocab_size, VIDEO_BEGIN_TOKEN)
    print(token_ids.shape)
    train_pipe = MDGPTTrainingPipeline(
        config=MDGPTTrainingPipeline.Config(
            log_dir=args.log_dir,
            project=args.project,
            vae_model_name=args.vae_model_name,
            vocab_size=vocab_size,
            meta_vocab_size=vocab_size,
            start_token=VIDEO_BEGIN_TOKEN,
            ignore_token=IGNORE_TOKEN,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
            sample_interval=args.sample_interval,
            batch_size=args.batch_size,
            block_size=args.block_size,
            sparse_block_size=args.sparse_block_size,
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
        )
    )
    train_pipe.fit(db)
    

if __name__ == "__main__":
    main()