from typing import List, Optional, Union, Dict
import torch
from torch import nn


class TorchTextEncoder(nn.Module):
    """A text encoder extracts a latent vector from text input.

    Similar to TorchImageEncoder, it wraps a backbone transformer model
    (e.g., BERT, RoBERTa, DistilBERT) with optional pooling and dense layers.

    The encoder takes text strings or pre-tokenized inputs and produces
    fixed-dimensional embeddings suitable for downstream tasks.

    Architecture:
        text → tokenizer → backbone → pooling → dense layers → dropout → embeddings

    The output should not directly be the model targets, but rather
    an intermediate representation for further processing.

    Example:
        >>> from .torch_text_backbone import create_text_backbone
        >>> backbone, tokenizer = create_text_backbone("distilbert-base-uncased")
        >>> encoder = TorchTextEncoder(
        ...     backbone=backbone,
        ...     tokenizer=tokenizer,
        ...     pool_name="cls",
        ...     dense_layer_sizes=[512],
        ... )
        >>> texts = ["This is a recipe", "Another recipe"]
        >>> embeddings = encoder(texts)  # Shape: (2, 512)
    """

    def __init__(
        self,
        name: str = "text_encoder",
        backbone: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
        pool_name: str = "cls",
        activation: str = "ReLU",
        dropout: float = 0.1,
        dense_layer_sizes: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        max_length: int = 512,
        finetune_backbone: bool = False,
    ):
        """Initialize text encoder.

        Args:
            name: Encoder name for identification
            backbone: HuggingFace transformer model (from create_text_backbone)
            tokenizer: HuggingFace tokenizer (from create_text_backbone)
            pool_name: Pooling method - "cls", "mean", or "max"
                - "cls": Use [CLS] token (first token) - good for BERT-style models
                - "mean": Mean pooling over all tokens - good for sentence embeddings
                - "max": Max pooling over all tokens
            activation: Activation function name (e.g., "ReLU", "GELU")
            dropout: Dropout probability
            dense_layer_sizes: Optional list of dense layer output sizes
            use_batch_norm: Whether to use batch normalization in dense layers
            max_length: Maximum sequence length for tokenization
            finetune_backbone: Whether to unfreeze backbone for fine-tuning
        """
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.pool_name = pool_name
        self.activation = activation
        self.dropout = dropout
        self.dense_layer_sizes = dense_layer_sizes
        self.use_batch_norm = use_batch_norm
        self.max_length = max_length
        self.finetune_backbone = finetune_backbone
        self.__post_init__()

    def __post_init__(self):
        assert self.backbone is not None, "Need a text backbone model."
        assert self.tokenizer is not None, "Need a tokenizer."

        # Optionally unfreeze backbone for fine-tuning
        if self.finetune_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True

        self._prepare_dropout_layers()
        self._prepare_dense_layers()
        self._activation = getattr(torch.nn, self.activation)()

    def forward(self, x: Union[List[str], str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through text encoder.

        Args:
            x: Either:
               - List of strings: ["text1", "text2", ...] (will be tokenized)
               - Single string: "text" (will be tokenized)
               - Dict with pre-tokenized inputs: {"input_ids": ..., "attention_mask": ...}

        Returns:
            Pooled text embedding tensor of shape (batch_size, embedding_dim)
        """
        # Handle tokenization
        if isinstance(x, (list, tuple, str)):
            # Text strings - tokenize them
            if isinstance(x, str):
                x = [x]
            elif isinstance(x, tuple):
                x = list(x)
            encoded = self.tokenizer(
                x,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = next(self.backbone.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        else:
            # Already tokenized - dict with input_ids and attention_mask
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]

        # Forward through backbone transformer
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pool token embeddings to fixed-size representation
        if self.pool_name == "cls":
            # Use [CLS] token (first token) - standard for BERT
            x = outputs.last_hidden_state[:, 0, :]
        elif self.pool_name == "mean":
            # Mean pooling over all tokens (considering attention mask)
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            x = torch.sum(token_embeddings * attention_mask_expanded, 1) / torch.clamp(
                attention_mask_expanded.sum(1), min=1e-9
            )
        elif self.pool_name == "max":
            # Max pooling over all tokens
            x = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:
            raise NotImplementedError(
                f"Pooling method '{self.pool_name}' not supported. "
                "Use 'cls', 'mean', or 'max'."
            )

        # Optional dense projection layers
        for d, batch_norm in zip(self._dense_layers, self._batch_norm_layers):
            x = d(x)
            if self.use_batch_norm:
                x = batch_norm(x)
            x = self._activation(x)

        x = self._dropout(x)
        return x

    def _prepare_dense_layers(self):
        """Prepare optional dense projection layers."""
        self._dense_layers = []
        self._batch_norm_layers = []
        for out_channels in self.dense_layer_sizes or []:
            _dense = nn.LazyLinear(out_channels)
            self._dense_layers.append(_dense)
            self._batch_norm_layers.append(nn.BatchNorm1d(num_features=out_channels))
        self._dense_layers = nn.ModuleList(self._dense_layers)
        self._batch_norm_layers = nn.ModuleList(self._batch_norm_layers)

    def _prepare_dropout_layers(self):
        """Prepare dropout layer."""
        self._dropout = nn.Dropout(self.dropout)


if __name__ == "__main__":
    from .torch_text_backbone import create_text_backbone

    print("Testing TorchTextEncoder...")

    # Create encoder with DistilBERT
    backbone, tokenizer = create_text_backbone("distilbert-base-uncased")
    encoder = TorchTextEncoder(
        backbone=backbone,
        tokenizer=tokenizer,
        pool_name="cls",
        dense_layer_sizes=[512],
        dropout=0.1,
    )

    # Test with text strings
    texts = [
        "2 cups flour, 1 egg, 1 cup milk",
        "Bake at 350F for 30 minutes",
        "Mix ingredients and serve cold",
    ]
    embeddings = encoder(texts)
    print(f"Input: {len(texts)} texts")
    print(f"Output shape: {embeddings.shape}")  # Should be (3, 512)
    print(f"Output dtype: {embeddings.dtype}")

    # Test with single string
    single_embedding = encoder("Test recipe")
    print(f"Single text output shape: {single_embedding.shape}")  # Should be (1, 512)
