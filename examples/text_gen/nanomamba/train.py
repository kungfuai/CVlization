from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from cvlization.torch.training_pipeline.lm.mamba import MambaTrainingPipeline
from cvlization.torch.training_pipeline.lm.data_utils import FlatTokenIds

parser = ArgumentParser()
parser.add_argument("--track", action="store_true")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--n_layer", type=int, default=12)
parser.add_argument("--pad_vocab_size_multiple", type=int, default=8)
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
with open(script_dir / "tinyshakespeare.txt", "r") as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
print("".join(chars))
vocab_size = len(chars)
print(vocab_size)

# Tokenizers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda xx: [stoi[x] for x in xx]
decode = lambda xx: "".join([itos[x] for x in xx])
encode("Hello!")
print(decode(encode("Hello!")))

token_ids = np.array(encode(text))
flat_token_ids = FlatTokenIds(
    token_ids, vocab_size=vocab_size + 1, start_token_id=None, train_split=0.8
)

train_pipe = MambaTrainingPipeline(
    config=MambaTrainingPipeline.Config(
        batch_size=args.batch_size,
        lr=args.lr,
        block_size=args.block_size,
        vocab_size=vocab_size + 1,
        d_model=args.d_model,
        n_layer=args.n_layer,
        pad_vocab_size_multiple=args.pad_vocab_size_multiple,
        device=args.device,
        decoder=decode,
        track=args.track,
    )
)
train_pipe.fit(flat_token_ids)

# torch.save(model.state_dict(), str(output_dir / "model.pt"))
# Generate from the model:
