import os
import numpy as np
import pickle
from cvlization.torch.training_pipeline.lm.gpt import NanoGPTTrainingPipeline

config_keys = NanoGPTTrainingPipeline.Config.__annotations__.keys()
# add the default values to the global namespace
example_config = NanoGPTTrainingPipeline.Config()
for k, v in NanoGPTTrainingPipeline.Config.__annotations__.items():
    if k in ["meta_vocab_size"]:
        continue
    default_value = example_config.__getattribute__(k)
    globals()[k] = default_value
exec(open("configurator.py").read())  # overrides from command line or config file
_globals = globals()
config = {
    k: _globals[k] for k in config_keys if k in _globals
}  # will be useful for logging
assert "dataset" in globals(), "Please specify the dataset to use"
# print(config)

# config["batch_size"] = 1
# config["learning_rate"] = 3e-4

# attempt to derive vocab_size from the dataset
data_dir = os.path.join("data", dataset)
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    config["itos"] = meta.get("itos")


def main():
    data_dir = os.path.join("data", dataset)
    use_program_aug = config.get("use_program_augmentation", False)

    if use_program_aug:
        pos_meta_path = os.path.join(data_dir, "meta_pos.pkl")
        assert os.path.exists(
            pos_meta_path
        ), f"Expected {pos_meta_path} when use_program_augmentation is True"
        with open(pos_meta_path, "rb") as f:
            meta_pos = pickle.load(f)
        config["program_offset"] = meta_pos["program_offset"]
        config["program_nil_id"] = meta_pos["program_nil_id"]
        config["program_vocab_size"] = len(meta_pos["program_pos_vocab"])
        config["program_pos_vocab"] = meta_pos.get("program_pos_vocab")
        train_bin = "train_with_pos.bin"
        val_bin = "val_with_pos.bin"
        dtype = np.uint32
    else:
        train_bin = "train.bin"
        val_bin = "val.bin"
        dtype = np.uint16

    train_data = np.memmap(os.path.join(data_dir, train_bin), dtype=dtype, mode="r")
    val_data = np.memmap(os.path.join(data_dir, val_bin), dtype=dtype, mode="r")

    class DatasetBuilder:
        def training_dataset(self):
            return train_data.astype(np.int32)

        def validation_dataset(self):
            return val_data.astype(np.int32)

    dataset_builder = DatasetBuilder()
    train_pipe = NanoGPTTrainingPipeline(
        NanoGPTTrainingPipeline.Config(meta_vocab_size=meta_vocab_size, **config)
    )
    train_pipe.fit(dataset_builder)


if __name__ == "__main__":
    main()
