from argparse import ArgumentParser
from cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline import TrainingPipeline


class MNISTDatasetBuilder:
    dataset_path = "./data/raw/MNIST"

    def training_dataset(self):
        return MNIST(root=self.dataset_path, train=True, download=True)

    def validation_dataset(self):
        return MNIST(root=self.dataset_path, train=False, download=True)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", "-d", type=str, default="mnist", help="dataset name")
    arg_parser.add_argument("--image_shape", "-s", type=str, help="image shape: e.g. 1,28,28", default="1,28,28")
    arg_parser.add_argument("--epochs", "-e", type=int, default=60, help="number of epochs")
    arg_parser.add_argument("--batch_size", "-b", type=int, default=128, help="batch size")
    args = arg_parser.parse_args()

    img_shape = tuple(map(int, args.image_shape.split(",")))
    if args.dataset.lower() == "mnist":
        from torchvision.datasets import MNIST

        dataset_builder = MNISTDatasetBuilder()
    elif "/" in args.dataset:
        from cvlization.data.huggingface import HuggingFaceDatasetBuilder

        if args.dataset in ["huggan/flowers-102-categories"]:
            validation_split_name = "train"
            img_shape = (3, img_shape[1], img_shape[2])
        else:
            validation_split_name = "validation"
        dataset_builder = HuggingFaceDatasetBuilder(
            args.dataset,
            validation_split_name=validation_split_name,
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    
    
    TrainingPipeline(
        img_shape=img_shape,
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
    ).fit(dataset_builder)