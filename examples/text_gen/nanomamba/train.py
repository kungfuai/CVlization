from pathlib import Path
import torch
import numpy as np
from cvlization.torch.training_pipeline.lm.mamba import MambaTrainingPipeline
from cvlization.torch.training_pipeline.lm.data_utils import FlatTokenIds

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
flat_token_ids = FlatTokenIds(token_ids, vocab_size=vocab_size+1, start_token_id=None, train_split=0.8)

train_pipe = MambaTrainingPipeline(
    config=MambaTrainingPipeline.Config(
        batch_size=32,
        lr=1e-3,
        block_size=128,
        vocab_size=512,
    )
)
train_pipe.fit(flat_token_ids)

# torch.save(model.state_dict(), str(output_dir / "model.pt"))
# Generate from the model:
device = train_pipe.config.device
output = train_pipe.model.generate(torch.zeros((1, 2), dtype=torch.long).to(device), 1000)[0].tolist()
