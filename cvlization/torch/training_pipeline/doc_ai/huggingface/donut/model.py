# pip install git+https://github.com/huggingface/transformers.git
# pip install datasets==2.4.* sentencepiece>=0.1.97 torchmetrics==0.9.* scikit-learn
import enum
import json
import logging
import numpy as np
import random
import re
from typing import Any, List
import torch
import pytorch_lightning as pl


LOGGER = logging.getLogger(__name__)


class DonutPredictionTask(enum.Enum):
    """Donut supports a variety of tasks, which are defined by the target type.
    """
    CLASSIFICATION = "classification"
    # DETECTION = "detection"


class DonutPLModule(pl.LightningModule):
    def __init__(self, processor, model, task=DonutPredictionTask.CLASSIFICATION):
        super().__init__()
        self.processor = processor
        self.model = model
        self.task = task
        # There is no info about number of classes in the Donut model! Because
        # its predicts a sequence that can be decoded to a class label.
        # self.train_accuracy = Accuracy(num_classes=10)

    def training_step(self, batch, batch_idx):
        model = self.model
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        processor = self.processor
        model = self.model
        pixel_values = batch["pixel_values"]
        
        # prepare decoder inputs
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(model.device)
        
        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        if self.task == DonutPredictionTask.CLASSIFICATION:
            scores = self._compute_classification_metrics(outputs, batch)
            return scores
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented!")
    
    def _compute_classification_metrics(self, outputs, batch):
        # turn into JSON
        processor = self.processor
        seqs = processor.batch_decode(outputs.sequences)
        labels = batch["labels"]
        ground_truths = [json.loads(g) for g in batch["ground_truth"]]
        scores = []
        for seq, label_token_ids, ground_truth in zip(seqs, labels, ground_truths):
            
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            seq = processor.token2json(seq)
            # print("seq:", seq)
            gt = ground_truth["gt_parse"]
            # print(seq.get("class"), gt["class"])
            score = float(seq.get("class") == gt["class"])
            scores.append(score)
        return scores
        
    def validation_epoch_end(self, validation_step_outputs):
        print(f"val_acc = {np.mean(validation_step_outputs)}  --------")
        self.log_dict({"val_acc": np.mean(validation_step_outputs)}, sync_dist=True)

    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class ProcessedDataset:
    # TODO: refactor this as a DatasetAdaptor (function that takes a dataset and returns a dataset)
    """
    ProcessedDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        processor: a processor which has tokenizer
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        source_dataset,
        processor,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.newly_added_num = 0

        self.dataset = source_dataset
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token] + (self.additional_tokens or []))
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    @property
    def additional_tokens(self):
        return ["<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", "<form/>", "<handwritten/>", "<invoice/>",
  "<letter/>", "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", "<resume/>",
  "<scientific_publication/>", "<scientific_report/>", "<specification/>"]

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.additional_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add tokens to tokenizer and resize the token embeddings of the decoder
        """
        processor = self.processor
        added_num = processor.tokenizer.add_tokens(list_of_tokens)
        self.newly_added_num += added_num
        if added_num > 0:
            LOGGER.info(f"Added {added_num} tokens to tokenizer. Total tokens: {len(processor.tokenizer)}")
        # TODO: the model's decoder also needs to be resized
        # if newly_added_num > 0:
        #     model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        processor = self.processor
        sample = self.dataset[idx]

        # pixel values (we remove the batch dimension)
        pixel_values = processor(sample["image"].convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # labels, which are the input ids of the target sequence
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        
        encoding = dict(
            pixel_values=pixel_values,
            labels=labels,
            ground_truth=sample["ground_truth"],
        )
        
        return encoding
