# pip install git+https://github.com/huggingface/transformers.git
# pip install datasets==2.4.* sentencepiece>=0.1.97 torchmetrics==0.9.* scikit-learn
import enum
import json
import logging
import numpy as np
import PIL
import random
import re
from typing import Any, List
import torch
from torch.utils.data import IterableDataset
import pytorch_lightning as pl
from nltk import edit_distance


LOGGER = logging.getLogger(__name__)


class DonutPredictionTask(enum.Enum):
    """Donut supports a variety of tasks, which are defined by the target type.
    """
    CLASSIFICATION = "classification"
    PARSE = "parse"
    CAPTION = "caption"


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

        is_train = True
        if labels is None:
            """
            Unseen/unlabeled example. We want loss of zero,
            so labels should be the prediction.
            """
            is_train = False
            token_predictions, text_predictions = self._predict_from_pixel_values(pixel_values=pixel_values)
            labels = token_predictions

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        if is_train:
            self.log_dict({"train_loss": loss}, sync_dist=True, on_step=True, prog_bar=True)
        else:
            # FIXME delete or use logger
            print(f"loss of {loss.item()} for unlabeled sample (prediction: {text_predictions[0]})")
            for ix, _fxn in enumerate(batch["update_fxn"]):
                _fxn(text_predictions[ix]) # TODO: Test this
        return loss

    def predict(self, image: PIL.Image):
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values
        token_predictions, text_predictions = self._predict_from_pixel_values(pixel_values=pixel_values)
        return text_predictions[0]

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        token_predictions, text_predictions = self._predict_from_pixel_values(pixel_values=batch["pixel_values"])

        return 0

        if self.task == DonutPredictionTask.CLASSIFICATION:
            return self._compute_classification_metrics(text_predictions, batch)
        elif self.task == DonutPredictionTask.PARSE:
            return self._compute_parse_metrics(text_predictions, batch)
        elif self.task == DonutPredictionTask.CAPTION:
            return self._compute_caption_metrics(text_predictions, batch)
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented!")

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        LOGGER.info(f"training_loss_epoch = {loss} --------")
        self.log_dict({"training_loss_epoch": loss}, sync_dist=True)

    def validation_epoch_end(self, validation_step_outputs):
        LOGGER.info(f"val_accuracy = {np.mean(validation_step_outputs)}  --------")
        self.log_dict({"val_accuracy": np.mean(validation_step_outputs)}, sync_dist=True)

    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def _predict_from_pixel_values(self, pixel_values):
        model, processor = self.model, self.processor

        # prepare decoder inputs
        task_prompt = processor.tokenizer.decode([model.config.decoder_start_token_id])
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

        """
        Copied from https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut
        """
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        return outputs.sequences, predictions

    def _compute_classification_metrics(self, predictions, batch):
        """
        # TODO: Untested code changes as of Jan 14, 2023
        """
        # turn into JSON
        processor = self.processor
        labels = batch["labels"]
        ground_truths = [json.loads(g) for g in batch["ground_truth"]]
        scores = []
        for pred, label_token_ids, ground_truth in zip(predictions, labels, ground_truths):
            seq = processor.token2json(pred)
            gt = ground_truth["gt_parse"]
            score = float(seq.get("class") == gt["class"])
            scores.append(score)
        return scores

    def _compute_parse_metrics(self, predictions, batch):
        """
        # TODO: Untested code changes as of Jan 14, 2023
        """
        scores = list()
        for pred, answer in zip(predictions, batch["target_sequence"]):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            score = 1 - edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)
            LOGGER.info(f"\nPrediction: \"{pred}\", Answer: \"{answer}\" (score: {score})")
        return scores

    def _compute_caption_metrics(self, predictions, batch):
        """
        Score by how many words the prediction and answer have in common.
        """
        scores = list()
        for pred, answer in zip(predictions, batch["target_sequence"]):
            # TODO: What does this regex do? Remove dataset task tokens?
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            # Remove <s_caption>...</s_caption> task tokens
            pred = pred.replace("<s_caption>", "").replace("</s_caption>", "")
            answer = answer.replace("<s_caption>", "").replace("</s_caption>", "")
            pred_words, answer_words = set(pred.split()), set(answer.split())
            # Now score
            common_words = pred_words.intersection(answer_words)
            common_words_total = 2 * len(common_words)
            total_words = len(pred_words) + len(answer_words)
            score = common_words_total / total_words
            scores.append(score)
            LOGGER.info(f"\nPredicted Caption: \"{pred}\", Answer: \"{answer}\" (score: {score})")
        return scores


class ProcessedDataset(IterableDataset):
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
        initialize_processor: bool = True,
        max_iterations: int = 100,
    ):
        super().__init__()

        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.max_iterations = max_iterations

        self.dataset = source_dataset
        if hasattr(self.dataset, '__len__'):
            self.dataset_length = len(self.dataset)
        else:
            self.dataset_length = 0 # Iterable

        if initialize_processor:
            """
            Process all dataset samples and add tokens to processor.tokenizer.
            Unnecessary if already performed and saved to disk.
            """
            self._initialize_processor()

        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
        is_class_token: bool = False):
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
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key, k == "class")
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        elif is_class_token:
            obj = str(obj)
            obj = f"<{obj}/>" # for categorical special tokens
            if self.add_tokens([obj]) > 0:
                """
                If processor was loaded from disk (instead of initialized using dataset),
                `self.additional_tokens` won't exist,
                and no new tokens should be added [during training].
                """
                self.additional_tokens.append(obj)
            return obj
        else:
            # Value
            return str(obj)

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add tokens to tokenizer and resize the token embeddings of the decoder
        """
        processor = self.processor
        added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if added_num > 0:
            LOGGER.info(f"Added {added_num} tokens to tokenizer. Total tokens: {len(processor.tokenizer)}")
        return added_num

    # def __len__(self) -> int:
    #     return self.dataset_length

    def __iter__(self):
        if self.dataset_length > 0:
            self._idx = 0
        else:
            self._iter = iter(self.dataset)
        self._iterations = 0
        return self

    def __next__(self) -> dict:
        self._iterations += 1
        if self._iterations > self.max_iterations:
            raise StopIteration
        if self.dataset_length > 0:
            return self._get_next_mapstyle()
        return self._get_next_iterstyle()

    def _initialize_processor(self):
        self.additional_tokens = []
        count = 0
        for sample in self.dataset:
            self._get_ground_truth_token_sequences(sample)
            count += 1
            if count >= 100:
                """
                Shouldn't need more than 100 samples to learn about all the tokens needed.
                """
                break
        self.add_tokens([self.task_start_token, self.prompt_end_token] + (self.additional_tokens or []))

    def _get_ground_truth_token_sequences(self, sample):
        """
        Get possible ground truth token sequences for a given sample index.
        """
        ground_truth = json.loads(sample["ground_truth"])
        if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            gt_jsons = [ground_truth["gt_parse"]]

        return [
            self.json2token(
                gt_json,
                update_special_tokens_for_json_key=self.split == "train",
                sort_json_key=self.sort_json_key,
            )
            + self.processor.tokenizer.eos_token
            for gt_json in gt_jsons  # load json from list of json
        ]

    def _get_next_mapstyle(self):
        """
        Get next single sample.
        """
        sample = self.dataset[self._idx]
        return self._encode_sample(sample)

    def _get_next_iterstyle(self):
        """
        Get next batch.
        """
        batch = next(self._iter)
        batch = [self._encode_sample(sample) for sample in batch]
        # Convert from list(dict()) to dict(list())
        batch = {
            key: [sample[key] for sample in batch]
            for key in batch[0].keys()
        }
        batch["pixel_values"] = torch.stack(batch["pixel_values"])
        batch["labels"] = torch.stack(batch["labels"]) if batch["labels"][0] is not None else None
        return batch

    def _encode_sample(self, sample):
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        processor = self.processor

        # pixel values (we remove the batch dimension)
        pixel_values = processor(sample["image"].convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        if sample["ground_truth"] is not None:
            # labels, which are the input ids of the target sequence
            gt_token_sequences = self._get_ground_truth_token_sequences(sample)
            target_sequence = random.choice(gt_token_sequences)  # can be more than one, e.g., DocVQA Task 1
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
        else:
            labels = None
            target_sequence = None

        encoding = dict(
            pixel_values=pixel_values,
            labels=labels,
            target_sequence=target_sequence,
            ground_truth=sample["ground_truth"],
            update_fxn=sample["update_fxn"] if "update_fxn" in sample else lambda: 0,
        )

        return encoding
