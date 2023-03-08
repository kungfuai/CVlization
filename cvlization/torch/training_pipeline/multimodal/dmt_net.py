"""Dynamic Modality and Task Network.
"""

from dataclasses import dataclass
import enum
from typing import Any, Union, List
import torch
from torch import nn
from timm.models.layers import trunc_normal_  # TODO: move to cvlization.
from .xdecoder_modules import MLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from .visual_text_dual_encoder import get_vision_model, get_text_model


class InvalidInputError(Exception):
    ...


class SizeVariant(str, enum.Enum):
    """A size variant of a modality."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"


@dataclass
class ModelInput:
    """A dataclass to hold the input to the model."""

    images: list = None  # support multiple images per training example
    text: str = None
    audio: Any = None
    video: Any = None
    dataframe: Any = None
    feature_dict: Any = None  # For tabular data
    # Supports free-formed tabular data rather than the user specifing
    # individual categorical and numerical columns.


@dataclass
class Example:
    """A dataclass to hold a training example."""

    input: ModelInput
    target: Any
    task: str


@dataclass
class DMTNet(nn.Module):
    """A transformer based architecture to support dynamic modality and task."""

    size_variant: SizeVariant = SizeVariant.SMALL
    num_classes: int = 10
    num_queries: int = 100
    num_layers: int = 6  # 12
    nheads: int = 8
    pre_norm: bool = False
    attention_hidden_dim: int = 512
    bbox_hidden_dim: int = 512
    dim_proj: int = 512
    dim_feedforward: int = 512  # 2048
    context_length: int = 77
    captioning_step: int = 50

    def __post_init__(self):
        super().__init__()
        self._build()

    def _build(self):
        # Encoders
        self.image_encoder, self.image_processor = get_vision_model()
        self.text_encoder, self.tokenizer = get_text_model()

        # Aggregators (general purpose reasoning on latents)
        nheads = self.nheads
        pre_norm = self.pre_norm
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.proj = nn.Linear(768, self.attention_hidden_dim)
        hidden_dim = self.attention_hidden_dim
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=self.dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        hidden_dim = self.bbox_hidden_dim
        # classification
        # self.class_embed = nn.Parameter(torch.empty(hidden_dim, self.dim_proj))
        # trunc_normal_(self.class_embed, std=0.02)
        # First pool the sequence of hidden states, and then dense.
        self.class_embed = nn.Sequential(
            nn.Linear(self.attention_hidden_dim, self.num_classes),
        )
        self.open_vocab_class_embed = None

        # object detection
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # image captioning
        self.caping_embed = nn.Parameter(torch.empty(hidden_dim, self.dim_proj))
        trunc_normal_(self.caping_embed, std=0.02)
        self.pos_embed_caping = nn.Embedding(self.context_length, hidden_dim)
        self.captioning_step = self.captioning_step

        # For sparse outputs (e.g. bounding boxes, segmentation masks)
        # From X-Decoder.
        self.num_queries = self.num_queries
        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        # From X-Decoder. Purpose??
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # register self_attn_mask to avoid information leakage, it includes interaction between object query, class query and caping query
        num_queries = self.num_queries
        contxt_len = self.context_length
        self_attn_mask = torch.zeros(
            (1, num_queries + contxt_len, num_queries + contxt_len)
        ).bool()
        self_attn_mask[
            :, :num_queries, num_queries:
        ] = True  # object+class query does not attend with caption query.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(
            torch.ones((1, contxt_len, contxt_len)), diagonal=1
        ).bool()  # caption query only attend with previous token.
        self_attn_mask[
            :, : num_queries - 1, num_queries - 1 : num_queries
        ] = True  # object query does not attend with class query.
        self_attn_mask[
            :, num_queries - 1 : num_queries, : num_queries - 1
        ] = True  # class query does not attend with object query.
        self.register_buffer("self_attn_mask", self_attn_mask)

    def forward(self, x: List[Example]):
        # The input could also be an Example type, with collated tensors.
        outputs = []
        for example in x:
            output = self.forward_single(example.input)
            outputs.append(output)
        return outputs

    def forward_single(
        self,
        x: Union[dict, ModelInput],
        output_mask: List[str] = None,
        input_mask: List[str] = None,
    ):
        x = ModelInput(**x) if isinstance(x, dict) else x
        x = self.select_input(x, input_mask=input_mask)
        encoded = self.encode(x)
        aggregated = self.aggregate(encoded)
        decoded = self.decode(aggregated, output_mask=output_mask)
        return {
            "encoded": encoded,
            "aggregated": aggregated,
            "output": decoded,
        }

    def forward_train(self, batch: List[Example]):
        # TODO: support collated examples.
        """A training step."""
        outputs = self.forward(batch)
        losses = []
        for output, example in zip(outputs, batch):
            loss = self.loss(output, example.target)
            losses.append(loss)
        loss_value = sum(losses)
        return loss_value

    def image_classification(self, x: ModelInput):
        return self.forward_single(x, output_mask=["category"], input_mask=["images"])

    def select_input(self, x: ModelInput, input_mask: List[str] = None):
        if isinstance(x, ModelInput):
            x = x.__dict__
        if input_mask is not None:
            x = {k: v for k, v in x.items() if k in input_mask}
        x = ModelInput(**x)
        return x

    def encode(self, x: dict):
        """Encoding. Embedding."""
        if isinstance(x, ModelInput):
            x = x.__dict__
        encoded = {}
        if x.get("images"):
            encoded["images"] = []

            # TODO: combine processor and model.forward as one encoder() callable.
            processed = self.image_processor(
                images=x["images"],
                return_tensors="pt",
                padding=True,
            )
            images_encoded = self.image_encoder(
                pixel_values=processed.pixel_values,
                output_hidden_states=True,
            )
            encoded["images"] = images_encoded  # typical transformer output.
        if x.get("text"):
            encoded["text"] = self.text_encoder(x.text)
        return encoded

    def aggregate(self, encoded: dict):
        """Fusing. Aggregation. Combine.

        Use self attention and cross attention to fuse the encoded features.
        """
        # prediction heads on learnable query features
        # print(encoded["images"][0])
        x = encoded["images"].hidden_states[-1]
        x = self.proj(x)
        return x[0][0]  # get the first image and the first token

    def decode(self, aggregated: dict, output_mask: List[str] = None):
        decoders = {
            "category": self.class_embed,
            "boxes": self.bbox_embed,
            "caption": self.caping_embed,
            # "mask": "TODO",
        }
        output = {}
        for output_name in output_mask or decoders.keys():
            output[output_name] = decoders[output_name](aggregated)
        return output

    def loss(self, output: dict, target: dict):
        pass


@dataclass
class DynamicModalityAndTaskPipeline:
    def fit(self, dataset_builder):
        pass


if __name__ == "__main__":
    import numpy as np

    model = DMTNet()
    img = np.random.rand(3, 224, 224)
    # TODO: support "image=" for a single image.
    x = ModelInput(images=[img])
    output = model.image_classification(x)  # similar interface as UnifiedIO
    print(output["output"]["category"])
    print("Done.")
