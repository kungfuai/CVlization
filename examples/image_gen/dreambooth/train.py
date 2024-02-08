import os
import gc
from pathlib import Path
import mimetypes
from dataclasses import dataclass
from argparse import ArgumentParser
import shutil
import sys
from dataclasses import field
import tempfile
from zipfile import ZipFile
from subprocess import call, check_call
from argparse import Namespace
import time
import torch

from .dreambooth import main


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def define_args():
    parser = ArgumentParser(description="Train Dreambooth model using example images.")
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a sks bird",
        help="The prompt you use to describe your training images, in the format: `a [identifier] [class noun]`, where the `[identifier]` should be a rare token. Relatively short sequences with 1-3 letters work the best (e.g. `sks`, `xjy`). `[class noun]` is a coarse class descriptor of the subject (e.g. cat, dog, watch, etc.). For example, your `instance_prompt` can be: `a sks dog`, or with some extra description `a photo of a sks dog`. The trained model will learn to bind a unique identifier with your specific subject in the `instance_data`.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a bird",
        help="The prompt or description of the coarse class of your training images, in the format of `a [class noun]`, optionally with some extra description. `class_prompt` is used to alleviate overfitting to your customised images (the trained model should still keep the learnt prior so that it can still generate different dogs when the `[identifier]` is not in the prompt). Corresponding to the examples of the `instant_prompt` above, the `class_prompt` can be `a dog` or `a photo of a dog`.",
    )
    parser.add_argument(
        "--instance_data",
        type=str,
        default="examples/image_gen/dreambooth/instance_data.zip",
        help="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
    )
    parser.add_argument(
        "--class_data",
        type=str,
        default=None,
        help="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=50,
        help="Minimal class images for prior preservation loss. If not enough images are provided in class_data, additional images will be sampled with class_prompt.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--pad_tokens",
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="Weight of prior preservation loss.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="A seed for reproducible training",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images. All the images in the train/validation dataset will be resized to this resolution.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        default="constant",
        help="The scheduler type to use",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="A list of concepts to use for training.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="The output directory where the logs will be written.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every n steps.",
    )
    parser.add_argument(
        "--hflip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "fp8", "bf16"],
        default="fp16",
        help="Whether or not to use mixed precision training.",
    )
    return parser


@dataclass
class DreamboothTrainingPipeline:
    args: Namespace = None

    def __post_init__(self):
        self.args = self.args or define_args().parse_args()
        self.setup()

    def setup(self):
        # check_call("nvidia-smi", shell=True)
        # assert torch.cuda.is_available()
        pass

    def fit(self) -> Path:
        args = self.args
        cog_instance_data = "cog_instance_data"
        cog_class_data = "cog_class_data"
        cog_output_dir = "checkpoints"
        for path in [cog_instance_data, cog_output_dir, cog_class_data]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # extract zip contents, flattening any paths present within it
        instance_data = Path(args.instance_data)
        class_data = Path(args.class_data) if args.class_data else None
        instance_prompt = args.instance_prompt
        class_prompt = args.class_prompt
        save_sample_prompt = args.save_sample_prompt
        save_sample_negative_prompt = args.save_sample_negative_prompt
        n_save_sample = args.n_save_sample
        save_guidance_scale = args.save_guidance_scale
        save_infer_steps = args.save_infer_steps
        pad_tokens = args.pad_tokens
        with_prior_preservation = args.with_prior_preservation
        prior_loss_weight = args.prior_loss_weight
        num_class_images = args.num_class_images
        seed = args.seed
        resolution = args.resolution
        center_crop = args.center_crop
        train_text_encoder = args.train_text_encoder
        train_batch_size = args.train_batch_size
        sample_batch_size = args.sample_batch_size
        num_train_epochs = args.num_train_epochs
        max_train_steps = args.max_train_steps
        gradient_accumulation_steps = args.gradient_accumulation_steps
        gradient_checkpointing = args.gradient_checkpointing
        learning_rate = args.learning_rate
        scale_lr = args.scale_lr
        lr_scheduler = args.lr_scheduler
        lr_warmup_steps = args.lr_warmup_steps
        use_8bit_adam = args.use_8bit_adam
        adam_beta1 = args.adam_beta1
        adam_beta2 = args.adam_beta2
        adam_weight_decay = args.adam_weight_decay
        adam_epsilon = args.adam_epsilon
        max_grad_norm = args.max_grad_norm
        output_dir = args.output_dir
        concepts_list = args.concepts_list
        logging_dir = args.logging_dir
        log_interval = args.log_interval
        hflip = args.hflip

        with ZipFile(str(instance_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_instance_data)

        if class_data is not None:
            with ZipFile(str(class_data), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, cog_class_data)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "pretrained_vae_name_or_path": "stabilityai/sd-vae-ft-mse",
            "revision": "fp16",
            "tokenizer_name": None,
            "instance_data_dir": cog_instance_data,
            "class_data_dir": cog_class_data,
            "instance_prompt": instance_prompt,
            "class_prompt": class_prompt,
            "save_sample_prompt": save_sample_prompt,
            "save_sample_negative_prompt": save_sample_negative_prompt,
            "n_save_sample": n_save_sample,
            "save_guidance_scale": save_guidance_scale,
            "save_infer_steps": save_infer_steps,
            "pad_tokens": pad_tokens,
            "with_prior_preservation": with_prior_preservation,
            "prior_loss_weight": prior_loss_weight,
            "num_class_images": num_class_images,
            "seed": seed,
            "resolution": resolution,
            "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "train_batch_size": train_batch_size,
            "sample_batch_size": sample_batch_size,
            "num_train_epochs": num_train_epochs,
            "max_train_steps": max_train_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": learning_rate,
            "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "use_8bit_adam": use_8bit_adam,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "push_to_hub": False,
            "hub_token": None,
            "hub_model_id": None,
            "save_interval": 10000,  # not used
            "save_min_steps": 0,
            "mixed_precision": args.mixed_precision,
            "not_cache_latents": False,
            "local_rank": -1,
            "output_dir": cog_output_dir,
            "concepts_list": None,
            "logging_dir": "logs",
            "log_interval": 10,
            "hflip": False,
        }

        args = Namespace(**args)

        main(args)

        gc.collect()
        # torch.cuda.empty_cache()
        # call("nvidia-smi")

        out_path = "output.zip"

        directory = Path(cog_output_dir)
        with ZipFile(out_path, "w") as z:
            for file_path in directory.rglob("*"):
                print(file_path)
                z.write(file_path, arcname=file_path.relative_to(directory))

        return Path(out_path)


if __name__ == "__main__":
    p = DreamboothTrainingPipeline()
    p.fit()
