import logging
from typing import Optional, Tuple
from torch import nn

LOGGER = logging.getLogger(__name__)


def load_huggingface_text_model(name: str, pretrained: bool = True) -> Tuple[nn.Module, object]:
    """Load a HuggingFace transformer model for text encoding.

    Args:
        name: HuggingFace model ID (e.g., "distilbert-base-uncased")
        pretrained: Whether to load pretrained weights (currently only True is supported)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ImportError: If transformers library is not installed
        NotImplementedError: If pretrained=False
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library is required for text encoders. "
            "Install with: pip install transformers"
        )

    if not pretrained:
        raise NotImplementedError("Only pretrained text models are supported")

    LOGGER.info(f"Loading HuggingFace model: {name}")

    # Load model and tokenizer
    model = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Note: Freezing is controlled by the freeze_text_encoder flag in ModelSpec
    # The TorchTextEncoder will handle freezing based on that flag

    LOGGER.info(f"Loaded {name} with {sum(p.numel() for p in model.parameters())} parameters")

    return model, tokenizer


def create_text_backbone(
    name: str,
    pretrained: bool = True,
) -> Tuple[nn.Module, object]:
    """Factory function to create text backbone models.

    This follows the same pattern as create_image_backbone, allowing easy
    model swapping by just changing the name string.

    Args:
        name: Model identifier. Supports any HuggingFace model ID.
        pretrained: Whether to load pretrained weights (default: True)

    Returns:
        Tuple of (model, tokenizer)

    Examples:
        >>> # DistilBERT (recommended - fast and efficient)
        >>> backbone, tokenizer = create_text_backbone("distilbert-base-uncased")

        >>> # BERT
        >>> backbone, tokenizer = create_text_backbone("bert-base-uncased")

        >>> # RoBERTa (better quality)
        >>> backbone, tokenizer = create_text_backbone("roberta-base")

        >>> # MiniLM (very fast)
        >>> backbone, tokenizer = create_text_backbone("microsoft/MiniLM-L6-v2")

        >>> # Sentence transformers (for similarity)
        >>> backbone, tokenizer = create_text_backbone("sentence-transformers/all-MiniLM-L6-v2")

    Supported models (partial list):
        - "distilbert-base-uncased" (recommended, 66M params, 768-dim)
        - "bert-base-uncased" (110M params, 768-dim)
        - "roberta-base" (125M params, 768-dim)
        - "microsoft/MiniLM-L6-v2" (23M params, 384-dim)
        - "sentence-transformers/all-MiniLM-L6-v2" (23M params, 384-dim)
        - Any HuggingFace transformer model with AutoModel support
    """
    # All models go through HuggingFace
    return load_huggingface_text_model(name, pretrained=pretrained)


if __name__ == "__main__":
    # Test loading different models
    print("Testing text backbone creation...")

    # Test DistilBERT
    model, tokenizer = create_text_backbone("distilbert-base-uncased")
    print(f"DistilBERT loaded: {type(model)}")

    # Test tokenization
    text = ["Hello world", "This is a recipe"]
    encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    print(f"Tokenized shape: {encoded['input_ids'].shape}")

    # Test forward pass
    outputs = model(**encoded)
    print(f"Output shape: {outputs.last_hidden_state.shape}")
