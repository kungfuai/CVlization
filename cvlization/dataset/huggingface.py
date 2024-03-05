from ..data.dataset_builder import Dataset


class HuggingfaceDatasetBuilder:
    """
    A wrapper for Huggingface's datasets library.
    """

    def __init__(
        self,
        dataset_name: str,
        train_split: str,
        val_split: str = None,
        test_split: str = None,
        load_dataset_args: dict = None,
    ):
        """
        Args:
            dataset_name: The name of the dataset to load.
            train_split: The name of the training split.
            val_split: The name of the validation split.
            test_split: The name of the test split.
            load_dataset_args: Additional arguments to pass to `load_dataset`. They will be passed as keyword arguments.
        """
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.huggingface_ds = load_dataset(dataset_name, **(load_dataset_args or {}))

    def training_dataset(self) -> Dataset:
        """
        Returns:
            The training dataset.
        """
        return self.huggingface_ds[self.train_split]

    def validation_dataset(self) -> Dataset:
        """
        Returns:
            The validation dataset.
        """
        return self.huggingface_ds[self.val_split]

    def test_dataset(self) -> Dataset:
        """
        Returns:
            The test dataset.
        """
        return self.huggingface_ds[self.test_split]
