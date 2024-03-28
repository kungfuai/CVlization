import os
import numpy as np
import pickle
from cvlization.torch.training_pipeline.lm.mdgpt import MDGPTTrainingPipeline

config_keys = MDGPTTrainingPipeline.Config.__annotations__.keys()
exec(open("configurator.py").read())  # overrides from command line or config file
_globals = globals()
config = {
    k: _globals[k] for k in config_keys if k in _globals
}  # will be useful for logging
assert "dataset" in globals(), "Please specify the dataset to use"
# print(config)

# attempt to derive vocab_size from the dataset
data_dir = os.path.join("data", dataset)
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


def main():
    data_dir = os.path.join("data", dataset)
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    class DatasetBuilder:
        def training_dataset(self):
            return train_data.astype(np.int32)  # [:50000]

        def validation_dataset(self):
            return val_data.astype(np.int32)  # [:50000]

    dataset_builder = DatasetBuilder()
    print(f"block_size = {block_size}")
    train_pipe = MDGPTTrainingPipeline(
        MDGPTTrainingPipeline.Config(
            vocab_size=meta_vocab_size + 5,
            meta_vocab_size=meta_vocab_size + 5,
            start_token=meta_vocab_size + 1,
            ignore_token=meta_vocab_size + 2,
            sparse_block_size=block_size,
            # n_layer=6,
            # n_head=6,
            # n_embd=384,
            **config,
            device="cuda",
            sample_interval=1000,
        )
    )
    train_pipe.fit(dataset_builder)


if __name__ == "__main__":
    main()
