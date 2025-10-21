from cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.argparser import parse_args
from cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.pipeline import TrainingPipeline, load_dataset


if __name__ == "__main__":
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    
    args = parse_args()

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    
    training_pipeline = TrainingPipeline(args)
    class DatasetBuilder:
        def training_dataset(self):
            return dataset
        
    training_pipeline.fit(DatasetBuilder())