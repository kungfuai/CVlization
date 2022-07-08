from dataclasses import dataclass


@dataclass
class LETRArgs:
    backbone: str = "resnet50"
    dilation: bool = False
    position_embedding: str = "sine"  # choices=("sine", "learned"),

    lr: float = 1e-4
    lr_backbone: float = 1e-5
    batch_size: int = 2
    epochs: int = 100
    weight_decay: float = 1e-4
    lr_drop: int = 50
    clip_max_norm: float = 0.1
    save_freq: int = 10

    # Train segmentation head if the flag is provided
    benchmark: bool = False
    # Name of the convolutional backbone to use
    append_word: str = None

    # Load
    layer1_frozen: bool = False
    layer2_frozen: bool = False
    # resume from checkpoint
    frozen_weights: str = ""
    resume: str = ""
    no_opt: bool = False

    # Transformer
    LETRpost: bool = False
    # layer1_num: int = 2
    layer1_num: int = 3
    layer2_num: int = 2

    # First Transformer
    # Number of encoding/decoding layers in the transformer
    enc_layers: int = 6
    dec_layers: int = 6
    # Intermediate size of the feedforward layers in the transformer blocks
    dim_feedforward: int = 2048
    # Size of the embeddings (dimension of the transformer)
    hidden_dim: int = 256
    # Dropout applied in the transformer
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 1000
    pre_norm: bool = False

    # Second Transformer
    second_enc_layers: int = 6
    second_dec_layers: int = 6
    second_dim_feedforward: int = 2048
    second_hidden_dim: int = 256
    second_dropout: float = 0.1
    second_nheads: int = 8
    second_pre_norm: bool = False

    # Loss
    aux_loss: bool = True

    # * Matcher
    # Class coefficient in the matching cost
    set_cost_class: float = 1
    # L1 box coefficient in the matching cost
    set_cost_line: int = 5
    # L1 box coefficient in the matching cost
    set_cost_point: int = 5

    # * Loss coefficients
    dice_loss_coef: float = 1
    point_loss_coef: float = 5
    line_loss_coef: float = 5
    eos_coef: float = 0.1
    label_loss_func: str = "cross_entropy"
    label_loss_params: str = "{}"

    # dataset parameters
    dataset_file: str = "coco"
    coco_path: str = None
    coco_panoptic_path: str = None
    remove_difficult: bool = False
    output_dir: str = ""

    device: str = "cuda"
    seed: int = 0
    start_epoch: int = 0
    num_workers: int = 2
    num_gpus: int = 1
