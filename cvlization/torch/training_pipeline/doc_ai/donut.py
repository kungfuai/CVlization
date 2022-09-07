from dataclasses import dataclass
import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor
from cvlization.torch.net.doc_ai.donut import DonutPLModule, ProcessedDataset, DonutPredictionTask


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
    max_epochs: int = 20
    accelerator: str = "gpu"
    devices: int = 1
    # For debugging
    limit_train_batches = None # 3
    limit_val_batches = None # 3

    def __post_init__(self):
        self._image_size = (self.image_height, self.image_width)  # TODO: reverse?
    
    def train(self, dataset_builder):
        config = self.create_config()
        trainer = self.create_trainer()
        processor = self.create_processor()
        train_dataloader, val_dataloader, newly_added_num = self.create_dataloaders(dataset_builder, processor)
        # newly_added_num: num of newly added tokens
        model = self.create_model(self.pretrained_model_name, newly_added_num, config, processor)
        pl_model = DonutPLModule(model=model, processor=processor, task=self.task)
        trainer.fit(pl_model, train_dataloader, val_dataloader)
    
    def create_config(self):
        config = VisionEncoderDecoderConfig.from_pretrained(self.pretrained_model_name)
        config.encoder.image_size = self._image_size
        config.decoder.max_length = self.max_length
        return config
    
    def create_processor(self):
        processor = DonutProcessor.from_pretrained(self.pretrained_model_name)
        processor.feature_extractor.size = self._image_size[::-1] # should be (width, height)
        processor.feature_extractor.do_align_long_axis = False
        return processor
    
    def process_dataset(self, dataset, processor):
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
    
    def create_model(self, pretrained_model_name: str, newly_added_num: int, config, processor):
        pretrained_model_name = self.pretrained_model_name
        model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name, config=config)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_rvlcdip>'])[0]
        return model
    
    def create_trainer(self):
        trainer = pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.max_epochs,
            limit_train_batches=self.limit_train_batches or 1.0,
            limit_val_batches=self.limit_val_batches or 1.0,
        )
        return trainer

    def create_dataloaders(self, dataset_builder, processor):
        train_dataset = self.process_dataset(dataset_builder.training_dataset(), processor)
        val_dataset = self.process_dataset(dataset_builder.validation_dataset(), processor)
        newly_added_num = train_dataset.newly_added_num
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_dataloader, val_dataloader, newly_added_num



