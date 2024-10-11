# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import wandb
import numpy as np
import tempfile
import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from cleanfid import fid

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./data/image_gen/cfm/results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

flags.DEFINE_string("logger", None, "Logger to use. Set to 'wandb' to use Weights & Biases")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    # Initialize wandb if specified
    if FLAGS.logger == 'wandb':
        project_name = os.getenv("WANDB_PROJECT") or "Diffuser Unconditional"
        wandb.init(project=project_name, config=FLAGS.flag_values_dict())

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                normal_samples = generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                ema_samples = generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                # calculate fid score
                # Save images to a temp directory
                temp_dir = tempfile.mkdtemp()
                save_dir = os.path.join(temp_dir, f"normal")
                for i, img in enumerate(normal_samples):
                    np.save(os.path.join(save_dir, f"{i}.npy"), img.cpu().numpy())
                fid_score_normal = fid.compute_fid(temp_dir, dataset_name='cifar10', dataset_res=32, mode='clean', batch_size=2)
                save_dir = os.path.join(temp_dir, f"ema")
                for i, img in enumerate(ema_samples):
                    np.save(os.path.join(save_dir, f"{i}.npy"), img.cpu().numpy())
                fid_score_ema = fid.compute_fid(temp_dir, dataset_name='cifar10', dataset_res=32, mode='clean', batch_size=2)
                print(f"FID score normal: {fid_score_normal}, FID score ema: {fid_score_ema}")
                
                # Log images to wandb
                if FLAGS.logger == 'wandb':
                    wandb.log({
                        "test_samples": [wandb.Image(img) for img in normal_samples[:8, ...]],
                        "test_samples_ema": [wandb.Image(img) for img in ema_samples[:8, ...]],
                        "step": step,
                        "fid_score_normal": fid_score_normal,
                        "fid_score_ema": fid_score_ema
                    })


            # Log metrics to wandb
            if FLAGS.logger == 'wandb':
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": sched.get_last_lr()[0],
                    "step": step
                })

    # Finish wandb run if it was initialized
    if FLAGS.logger == 'wandb':
        wandb.finish()


if __name__ == "__main__":
    app.run(train)
