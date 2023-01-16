from pathlib import Path
import logging
from typing import Union

from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import transformers
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor
from .model import DonutPLModule, ProcessedDataset, DonutPredictionTask


LOGGER = logging.getLogger(__name__)


@dataclass
class Donut:
    """Donut is a TrainingPipeline. It is a task-agnositic model, with output
    being sequences that can decode into a variety of target types (e.g. classification label,
    bounding boxes etc.).
    """
    pretrained_model_name: str = "nielsr/donut-base"
    task: DonutPredictionTask = DonutPredictionTask.CLASSIFICATION
    max_length: int = 8
    # image_size = [2560, 1920]  # smaller image size has lower accuracy, though training is faster
    image_height: int = 1024
    image_width: int = 768
    ignore_id: int = -100
    task_start_token: str = "<s>"
    prompt_end_token: str = None
    sort_json_key: bool = True
    max_epochs: int = 100
    accelerator: str = "gpu"
    devices: int = 1
    # For debugging
    limit_train_batches = None # 3
    limit_val_batches = None # 3

    def __post_init__(self):
        self._image_size = (self.image_height, self.image_width)  # TODO: reverse?
    
    def fit(self, dataset_builder):
        self.train(dataset_builder)

    def eval(self):
        config = self._create_config()
        processor = self._load_latest_processor_or_create()
        model = self._create_model(self.pretrained_model_name, config, processor)
        pl_model = self._load_latest_pl_model_or_create(model=model, processor=processor)
        return pl_model

    def train(self, dataset_builder):
        config = self._create_config()
        trainer = self._create_trainer()
        processor = self._create_processor() # TODO: Refactor to use cached processor.
        # newly_added_num: num of newly added tokens
        train_dataloader, val_dataloader, newly_added_num = self._create_dataloaders(dataset_builder, processor)
        model = self._create_model(self.pretrained_model_name, config, processor)
        pl_model = self._load_latest_pl_model_or_create(model=model, processor=processor)
        # Save the modified processor for use in inference
        # (Do this after looking for latest experiment directory because a new one will be created.)
        processor_save_dir = Path("lightning_logs") / f"version_{trainer.logger.version}" / "processor"
        processor_save_dir.mkdir(exist_ok=True, parents=True)
        processor.save_pretrained(processor_save_dir)
        trainer.fit(pl_model, train_dataloader, val_dataloader)

    def _create_config(self):
        config = VisionEncoderDecoderConfig.from_pretrained(self.pretrained_model_name)
        config.encoder.image_size = self._image_size
        config.decoder.max_length = self.max_length
        return config

    def _create_processor(self):
        processor = DonutProcessor.from_pretrained(self.pretrained_model_name)
        processor.feature_extractor.size = self._image_size[::-1] # should be (width, height)
        processor.feature_extractor.do_align_long_axis = False
        return processor

    def _process_dataset(self, dataset, processor):
        # prepare the dataset using the Processor
        return ProcessedDataset(
            source_dataset=dataset,
            processor=processor,
            max_length=self.max_length,
            ignore_id=self.ignore_id,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
            sort_json_key=self.sort_json_key,
        )

    def _create_model(self, pretrained_model_name: str, config, processor):
        pretrained_model_name = self.pretrained_model_name
        model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name, config=config)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([
            self.task_start_token,
        ])[0]
        return model

    def _create_trainer(self):
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            # strategy="ddp",
            max_epochs=self.max_epochs,
            limit_train_batches=self.limit_train_batches or 1.0,
            limit_val_batches=self.limit_val_batches or 1.0,
            accumulate_grad_batches=2,
        )
        return trainer

    def _create_dataloaders(self, dataset_builder, processor):
        train_dataset = self._process_dataset(dataset_builder.training_dataset(), processor)
        val_dataset = self._process_dataset(dataset_builder.validation_dataset(), processor)
        newly_added_num = train_dataset.newly_added_num
        batch_size = 1
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, val_dataloader, newly_added_num

    def _get_latest_experiment_version(self) -> Union[int, None]:
        experiment_versions = [
            int(str(exp_dir.name).split("_")[1])
            for exp_dir in self.llogs.iterdir()
            if str(exp_dir.name).startswith("version_")
        ]
        if len(experiment_versions) == 0:
            return None
        return max(experiment_versions)

    def _get_latest_experiment_dir(self) -> Union[Path, None]:
        latest_version = self._get_latest_experiment_version()
        return self.llogs / f"version_{latest_version}" if latest_version is not None else None

    def _get_experiment_subdir(self, subdir_name: str) -> Union[Path, None]:
        exp_dir = self._get_latest_experiment_dir()
        subdir = exp_dir / subdir_name if exp_dir is not None else None
        return subdir if subdir is not None and subdir.exists() else None

    def _load_latest_processor_or_create(self):
        processor_dir = self._get_experiment_subdir("processor")
        return DonutProcessor.from_pretrained(str(processor_dir)) \
            if processor_dir is not None else self._create_processor()

    def _load_latest_pl_model_or_create(self, model, processor):
        # Auto-find latest checkpoint if exists (should only be 1 checkpoint from latest experiment)
        # Directory existing implies not empty (UNLESS first epoch didn't finish, in which case, delete the directory)
        # TODO: Could be more robust
        checkpoints_dir = self._get_experiment_subdir("checkpoints")
        checkpoints = list(checkpoints_dir.iterdir()) if checkpoints_dir is not None else []
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0] # Assume only 1 checkpoint saved.
            LOGGER.info(f"Found model checkpoint at \"{checkpoint}\".")
            return DonutPLModule.load_from_checkpoint(checkpoint, model=model, processor=processor, task=self.task)
        LOGGER.info("Could not find model checkpoint.")
        return DonutPLModule(model=model, processor=processor, task=self.task)

    @property
    def llogs(self):
        return Path("lightning_logs")
