def main():
    from datasets import Dataset, DatasetDict

    # from huggingface_hub import notebook_login
    from .flying_mnist import FlyingMNISTDatasetBuilder

    # notebook_login()

    db = FlyingMNISTDatasetBuilder()
    train_ds = db.training_dataset()
    val_ds = db.validation_dataset()

    train_ds = Dataset.from_list(train_ds)
    val_ds = Dataset.from_list(val_ds)

    hf_dataset = DatasetDict(
        {
            "train": train_ds,
            "val": val_ds,
        }
    )
    hf_dataset.push_to_hub(repo_id="zzsi/flying_mnist", max_shard_size="1GB")


if __name__ == "__main__":
    main()
