import argparse
import logging
import torch

from cvlization.dataset.recipe_dataset import RecipeDatasetBuilder
from cvlization.torch.torch_training_pipeline import TorchTrainingPipeline

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal recipe model (image + tabular + text).")
    parser.add_argument("--dataset-id", type=str, default="zzsi/recipes_10k", help="Hugging Face dataset id to load.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Split name within the dataset.")
    parser.add_argument("--max-examples", type=int, default=8000, help="Maximum number of recipes to use.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="resnet18", help="Image backbone (resnet18, resnet50, etc.)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps-per-epoch", type=int, default=None, help="Optional cap on train batches per epoch.")
    parser.add_argument("--val-steps-per-epoch", type=int, default=None, help="Optional cap on validation batches per epoch.")

    # Text encoder options
    parser.add_argument("--use-text", action="store_true", default=True, help="Use text features (ingredients)")
    parser.add_argument("--no-text", dest="use_text", action="store_false", help="Disable text features")
    parser.add_argument("--text-backbone", type=str, default="distilbert-base-uncased",
                        help="Text encoder backbone (distilbert-base-uncased, bert-base-uncased, roberta-base, etc.)")
    parser.add_argument("--text-pool", type=str, default="cls", choices=["cls", "mean", "max"],
                        help="Text pooling method")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")

    LOGGER.info("Loading recipe dataset %s[%s] from Hugging Face…", args.dataset_id, args.dataset_split)
    LOGGER.info("Text features: %s (backbone: %s)", "enabled" if args.use_text else "disabled", args.text_backbone if args.use_text else "N/A")

    dataset_builder = RecipeDatasetBuilder(
        dataset_name=args.dataset_id,
        dataset_split=args.dataset_split,
        max_examples=args.max_examples,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        image_size=args.image_size,
        pretrained_backbone=args.backbone,
        use_text_features=args.use_text,
        text_backbone=args.text_backbone,
        text_pool_method=args.text_pool,
    )
    LOGGER.info(
        "Dataset prepared with %d training and %d validation examples.",
        len(dataset_builder._train_indices),
        len(dataset_builder._val_indices),
    )

    if args.use_text:
        LOGGER.info("Model architecture: Tri-modal (image + tabular + text)")
    else:
        LOGGER.info("Model architecture: Bi-modal (image + tabular)")

    model_spec = dataset_builder.build_model_spec()
    pipeline = TorchTrainingPipeline(
        model=model_spec,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        collate_method=None,
        log_every_n_steps=10,
        train_steps_per_epoch=args.train_steps_per_epoch,
        val_steps_per_epoch=args.val_steps_per_epoch,
    )

    pipeline.fit(dataset_builder)


if __name__ == "__main__":
    main()
