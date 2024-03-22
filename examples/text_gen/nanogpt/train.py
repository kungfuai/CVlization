import os
import numpy as np
import pickle
from cvlization.torch.training_pipeline.lm.gpt import NanoGPTTrainingPipeline

config_keys = NanoGPTTrainingPipeline.Config.__annotations__.keys()
exec(open("configurator.py").read())  # overrides from command line or config file
_globals = globals()
config = {k: _globals[k] for k in config_keys if k in _globals}  # will be useful for logging
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
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    class DatasetBuilder:
        def training_dataset(self):
            return train_data.astype(np.int32)
        
        def validation_dataset(self):
            return val_data.astype(np.int32)
    
    dataset_builder = DatasetBuilder()
    train_pipe = NanoGPTTrainingPipeline(NanoGPTTrainingPipeline.Config(meta_vocab_size=meta_vocab_size, **config))
    train_pipe.fit(dataset_builder)


if __name__ == "__main__":
    main()