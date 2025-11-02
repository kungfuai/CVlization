import ast
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from ..data.dataset_builder import BaseDatasetBuilder, DatasetProvider
from ..specs import (
    DataColumnType,
    ModelInput,
    ModelSpec,
    ModelTarget,
)
from ..specs.losses.loss_type import LossType


TIME_MULTIPLIERS = {
    "day": 24 * 60,
    "days": 24 * 60,
    "d": 24 * 60,
    "hour": 60,
    "hours": 60,
    "hr": 60,
    "hrs": 60,
    "h": 60,
    "minute": 1,
    "minutes": 1,
    "min": 1,
    "mins": 1,
    "m": 1,
    "second": 1 / 60,
    "seconds": 1 / 60,
    "sec": 1 / 60,
    "secs": 1 / 60,
    "s": 1 / 60,
}

LOGGER = logging.getLogger(__name__)


def _parse_time_to_minutes(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    value = str(value).strip().lower()
    if not value:
        return None
    total_minutes = 0.0
    for number, unit in re.findall(r"(\d+)\s*([a-z]+)", value):
        multiplier = TIME_MULTIPLIERS.get(unit)
        if multiplier is None:
            continue
        total_minutes += int(number) * multiplier
    return total_minutes if total_minutes > 0 else None


def _safe_literal_eval(value: Optional[str]) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        raise TypeError(
            f"Expected ingredient/instruction field to be str or sequence, got {type(value)} with value {value!r}"
        )
    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return parsed
    except Exception:
        pass
    # Fallback: split by semicolon or newline.
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    if "\n" in value:
        return [item.strip() for item in value.split("\n") if item.strip()]
    return [value]


def _coerce_float(value, default: float = 0.0, clamp_min: Optional[float] = None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    if clamp_min is not None:
        numeric = max(numeric, clamp_min)
    return numeric


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))


def _ensure_finite_tensor(name: str, tensor: torch.Tensor, example) -> None:
    if not torch.isfinite(tensor).all():
        title = example.get("title", "<unknown>")
        raise ValueError(
            f"Non-finite values detected in {name} for example '{title}'. "
            f"Tensor contents: {tensor}"
        )


class RecipeTorchDataset(torch.utils.data.Dataset):
    def __init__(self, builder: "RecipeDatasetBuilder", indices: Sequence[int]):
        self.builder = builder
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        example = self.builder.dataset[self.indices[idx]]
        processed = self.builder.process_example(example)

        if self.builder.use_text_features:
            image_tensor, feature_tensor, text_data, targets = processed
            calories_tensor, total_time_tensor = targets
            return [image_tensor, feature_tensor, text_data], [calories_tensor, total_time_tensor]
        else:
            image_tensor, feature_tensor, targets = processed
            calories_tensor, total_time_tensor = targets
            return [image_tensor, feature_tensor], [calories_tensor, total_time_tensor]


@dataclass
class RecipeDatasetBuilder(BaseDatasetBuilder):
    """Prepare multimodal recipe data for torch training.

    Supports tri-modal learning: image + tabular features + text.
    """

    dataset_name: str = "zzsi/recipes_10k"
    dataset_split: str = "train"
    max_examples: Optional[int] = 8000
    val_ratio: float = 0.2
    random_seed: int = 42
    image_size: int = 224
    pretrained_backbone: str = "resnet18"

    # Text encoder configuration
    use_text_features: bool = True
    text_backbone: str = "distilbert-base-uncased"
    text_max_length: int = 256  # Shorter than default 512 for efficiency
    text_pool_method: str = "cls"  # "cls", "mean", or "max"
    text_dense_layers: Optional[List[int]] = None  # e.g., [512] for projection

    # Encoder freezing configuration
    freeze_image_encoder: bool = True
    freeze_text_encoder: bool = True

    # Data validation (can add ~2min overhead for 6k examples)
    # When False, relies on PyTorch DataLoader to catch errors during training
    validate_data: bool = False

    dataset_provider: DatasetProvider = DatasetProvider.CVLIZATION

    def __post_init__(self):
        # Load the specific split directly
        self.dataset = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
        )
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.validate_data:
            # Validate and filter examples (adds ~2min for 6k examples)
            LOGGER.info("Data validation enabled - filtering invalid examples")
            valid_indices: List[int] = []
            dropped_by_reason: dict[str, int] = {}

            # Determine how many examples we need to validate
            # Add buffer for validation failures (e.g., 50% more than needed)
            max_to_validate = len(self.dataset)
            if self.max_examples is not None:
                # Add 50% buffer to account for validation failures
                max_to_validate = min(int(self.max_examples * 1.5), len(self.dataset))

            for idx in range(max_to_validate):
                example = self.dataset[idx]
                reason = self._validate_example(example)
                if reason is None:
                    valid_indices.append(idx)
                    # Early stopping: stop once we have enough valid examples
                    if self.max_examples is not None and len(valid_indices) >= self.max_examples:
                        LOGGER.info(f"Collected {len(valid_indices)} valid examples (target: {self.max_examples}), stopping validation early")
                        break
                else:
                    dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1

            if self.max_examples is not None:
                valid_indices = valid_indices[: self.max_examples]

            if dropped_by_reason:
                summary = ", ".join(f"{reason}={count}" for reason, count in dropped_by_reason.items())
                LOGGER.warning(
                    "Dropped %d examples due to validation failures: %s",
                    sum(dropped_by_reason.values()),
                    summary,
                )
        else:
            # Skip validation - use all examples (or first max_examples)
            LOGGER.info("Data validation disabled - using examples as-is (faster, errors caught during training)")
            n_examples = self.max_examples if self.max_examples is not None else len(self.dataset)
            n_examples = min(n_examples, len(self.dataset))
            valid_indices = list(range(n_examples))

        # Shuffle and split into train/val
        rng = np.random.default_rng(self.random_seed)
        rng.shuffle(valid_indices)
        split_index = max(1, int(len(valid_indices) * (1 - self.val_ratio)))
        self._train_indices = valid_indices[:split_index]
        self._val_indices = valid_indices[split_index:]

        if len(self._val_indices) == 0:
            raise ValueError("Validation split is empty. Increase max_examples or adjust val_ratio.")

        # Determine feature dimension using the first valid example.
        sample_example = self.dataset[self._train_indices[0]]
        processed = self.process_example(sample_example)
        if self.use_text_features:
            _, feature_tensor, _, _ = processed
        else:
            _, feature_tensor, _ = processed
        self.feature_dim = feature_tensor.shape[-1]

    def training_dataset(self) -> torch.utils.data.Dataset:
        return RecipeTorchDataset(self, self._train_indices)

    def validation_dataset(self) -> torch.utils.data.Dataset:
        return RecipeTorchDataset(self, self._val_indices)

    def build_model_spec(self) -> ModelSpec:
        # Build model inputs
        model_inputs = [
            ModelInput(
                key="image",
                column_type=DataColumnType.IMAGE,
                raw_shape=[self.image_size, self.image_size, 3],
            ),
            ModelInput(
                key="recipe_features",
                column_type=DataColumnType.NUMERICAL,
                raw_shape=[self.feature_dim],
            ),
        ]

        # Add text input if enabled
        if self.use_text_features:
            from ..torch.encoder.torch_text_backbone import create_text_backbone
            from ..torch.encoder.torch_text_encoder import TorchTextEncoder

            # Create text encoder
            LOGGER.info(f"Creating text encoder with backbone: {self.text_backbone}")
            backbone, tokenizer = create_text_backbone(self.text_backbone)

            # Determine output dimension (backbone dim or last dense layer)
            if self.text_dense_layers:
                text_output_dim = self.text_dense_layers[-1]
            else:
                # Get embedding dimension from a test forward pass
                test_encoded = tokenizer(["test"], return_tensors="pt", max_length=self.text_max_length, truncation=True)
                test_output = backbone(**test_encoded)
                text_output_dim = test_output.last_hidden_state.shape[-1]

            text_encoder = TorchTextEncoder(
                backbone=backbone,
                tokenizer=tokenizer,
                pool_name=self.text_pool_method,
                dense_layer_sizes=self.text_dense_layers,
                max_length=self.text_max_length,
                dropout=0.1,
                finetune_backbone=(not self.freeze_text_encoder),
            )

            model_inputs.append(
                ModelInput(
                    key="ingredients",
                    column_type=DataColumnType.TEXT,
                    raw_shape=[text_output_dim],
                )
            )

            # Store text encoder to be passed to model
            self._text_encoder = text_encoder
        else:
            self._text_encoder = None

        return ModelSpec(
            model_inputs=model_inputs,
            model_targets=[
                ModelTarget(
                    key="calories",
                    column_type=DataColumnType.NUMERICAL,
                    raw_shape=[1],
                    loss=LossType.MSE,
                ),
                ModelTarget(
                    key="total_time_minutes",
                    column_type=DataColumnType.NUMERICAL,
                    raw_shape=[1],
                    loss=LossType.MSE,
                ),
            ],
            image_backbone=self.pretrained_backbone,
            provider="torchvision",
            pretrained=True,
            permute_image=True,
            input_shape=[self.image_size, self.image_size, 3],
            dense_layer_sizes=[512],
            freeze_image_encoder=self.freeze_image_encoder,
            freeze_text_encoder=self.freeze_text_encoder,
            text_encoder=self._text_encoder,  # Pass text encoder to model
        )

    # Processing -----------------------------------------------------------------
    def process_example(self, example):
        image_tensor = self._prepare_image(example["image"])
        feature_tensor = self._build_feature_tensor(example)

        # Prepare text data if enabled
        if self.use_text_features:
            text_data = self._prepare_text(example)

        calories_value = _coerce_float(example.get("calories"), default=float("nan"))
        if math.isnan(calories_value):
            title = example.get("title", "<unknown>")
            raise ValueError(f"Non-finite calories for example '{title}': {example.get('calories')}")
        total_minutes = self._get_total_minutes(example)
        if total_minutes is None or math.isnan(total_minutes):
            title = example.get("title", "<unknown>")
            raise ValueError(f"Non-finite total_time for example '{title}': {example.get('total_time')}")
        calories_tensor = torch.tensor([calories_value], dtype=torch.float32)
        total_time_tensor = torch.tensor([total_minutes], dtype=torch.float32)
        _ensure_finite_tensor("feature tensor", feature_tensor, example)
        _ensure_finite_tensor("calories", calories_tensor, example)
        _ensure_finite_tensor("total_time_minutes", total_time_tensor, example)

        if self.use_text_features:
            return image_tensor, feature_tensor, text_data, (calories_tensor, total_time_tensor)
        else:
            return image_tensor, feature_tensor, (calories_tensor, total_time_tensor)

    def _prepare_image(self, image) -> torch.Tensor:
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            image = Image.open(image_path)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(
                f"Unsupported image type {type(image)}; expected PIL.Image.Image, numpy.ndarray, or path string."
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_array = np.asarray(image).astype(np.float32) / 255.0
        image_array = (image_array - self._mean) / self._std
        return torch.tensor(image_array, dtype=torch.float32)

    def _build_feature_tensor(self, example) -> torch.Tensor:
        counts = _safe_literal_eval(example.get("ingredients"))
        instructions = _safe_literal_eval(
            example.get("instructions_list") or example.get("instructions")
        )
        ingredient_count = float(len(counts))
        instruction_count = float(len(instructions))
        rating = _coerce_float(example.get("rating"), default=0.0)
        rating_count = _coerce_float(example.get("rating_count"), default=0.0, clamp_min=0.0)
        review_count = _coerce_float(example.get("review_count"), default=0.0, clamp_min=0.0)
        servings = _coerce_float(example.get("servings"), default=0.0, clamp_min=0.0)
        prep_minutes = self._get_prep_minutes(example)
        cook_minutes = self._get_cook_minutes(example)

        macro_keys = [
            "carbohydrates_g",
            "sugars_g",
            "fat_g",
            "saturated_fat_g",
            "protein_g",
            "dietary_fiber_g",
            "sodium_mg",
        ]
        macro_values = [
            _coerce_float(example.get(key), default=0.0)
            for key in macro_keys
        ]

        vector = [
            rating,
            _safe_log1p(rating_count),
            _safe_log1p(review_count),
            servings,
            ingredient_count,
            instruction_count,
            prep_minutes,
            cook_minutes,
        ]
        vector.extend(macro_values)
        feature_array = np.asarray(vector, dtype=np.float32)
        return torch.tensor(feature_array, dtype=torch.float32)

    def _prepare_text(self, example) -> str:
        """Prepare text representation of recipe (ingredients list).

        The text will be tokenized and encoded by the text encoder in the model.

        Returns:
            A string containing the ingredients list, e.g.,
            "2 cups flour, 1 egg, 1 cup milk, 1/2 tsp salt"
        """
        ingredients = _safe_literal_eval(example.get("ingredients"))

        if not ingredients:
            # Fallback to empty string if no ingredients
            return ""

        # Join ingredients into a comma-separated string
        # This preserves the natural language structure for the text encoder
        if isinstance(ingredients, list):
            # Filter out empty strings and convert to strings
            ingredients_str = [str(ing).strip() for ing in ingredients if ing]
            text = ", ".join(ingredients_str)
        else:
            text = str(ingredients)

        # Optionally prepend a prompt to give context (helps with transformer understanding)
        text = f"Ingredients: {text}"

        return text

    def _validate_example(self, example) -> Optional[str]:
        calories = example.get("calories")
        if calories is None:
            return "missing_calories"
        if isinstance(calories, (float, int)) and (math.isnan(float(calories)) or math.isinf(float(calories))):
            return "nonfinite_calories"

        total_minutes = self._get_total_minutes(example)
        if total_minutes is None:
            return "missing_total_time"

        image = example.get("image")
        if image is None:
            return "missing_image"
        try:
            # Lightweight image validation - just check if it can be opened
            # Don't do expensive resize/normalize operations during validation
            if isinstance(image, str):
                # For file paths, just verify the image can be opened
                with Image.open(image) as img:
                    # Quick check: ensure it's a valid image format
                    _ = img.size  # Access size to ensure header is readable
            elif isinstance(image, np.ndarray):
                # Already in memory as array, valid
                pass
            elif isinstance(image, Image.Image):
                # Already a PIL Image, valid
                pass
            else:
                return "invalid_image_type"
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Invalid image for %s: %s", example.get("title", "<unknown>"), exc)
            return "invalid_image"

        try:
            feature_tensor = self._build_feature_tensor(example)
            _ensure_finite_tensor("feature tensor", feature_tensor, example)
        except ValueError:
            return "nonfinite_features"

        return None

    # Helpers -------------------------------------------------------------------
    @staticmethod
    def _maybe_get_minutes(example, candidate_keys: Sequence[str]) -> Optional[float]:
        for key in candidate_keys:
            if key not in example:
                continue
            value = example.get(key)
            if value is None:
                continue
            minutes = _parse_time_to_minutes(value)
            if minutes is not None and not (math.isnan(minutes) or math.isinf(minutes)):
                return max(minutes, 0.0)
        return None

    def _get_total_minutes(self, example) -> Optional[float]:
        return self._maybe_get_minutes(example, ("total_time_minutes", "total_time"))

    def _get_prep_minutes(self, example) -> float:
        minutes = self._maybe_get_minutes(example, ("prep_time_minutes", "prep_time"))
        return minutes if minutes is not None else 0.0

    def _get_cook_minutes(self, example) -> float:
        minutes = self._maybe_get_minutes(example, ("cook_time_minutes", "cook_time"))
        return minutes if minutes is not None else 0.0
