"""
Created on Tue Jun 14 13:43:43 2022
@author: zhuya
"""
# try to reproduce ruiqi's T6 model

import numpy as np
import os
import torch
import random
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import datetime as dt
import re
import shutil
import math
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
import pytorch_fid_wrapper as pfw
import torch.multiprocessing
import wandb
from .ebm2 import EBM_notemb, EBM

wandb.init(project="cdrl")

torch.multiprocessing.set_sharing_strategy("file_system")

# implement diffusion ebm with cooperative learning

########################## hyper parameters ###################################
tpu = False
seed = 3
batch_size = 64
num_workers = 4
c = 3
im_sz = 32
n_interval = 6
n_updates = 30
n_blocks = 8
logsnr_min = -5.1
logsnr_max = 9.8
latent_dim = 100
Langevin_clip = False

log_path = "./logs/cifar10"
training = True


ckpt_idx = None  # 200000
load_path = None  #'./logs/cifar10/20221017_125151/ckpt/{}.pth.tar'.format(ckpt_idx)

pi_lr = 1e-4
warmup_steps = 10000
iterations = 1000001
pi_ema_decay = 0.9999
grad_clip = 1.0
add_q = False
with_sn = True  # whether use spectral norm in model
sn_step = 1
ebm_act = "lrelu"
pred_var_type = "small"

max_pretrain_pi_iter = 100001
print_iter = 1
plot_iter = 500
ckpt_iter = 50000
fid_iter = 25000
n_fid_samples = 50000
n_fid_samples_full = 50000

reduce_variance = True  # whether use reduce variance technique

eta = 1.0  # add a coefficient to rescale the ratio term --> enable gradient to flow through

wandb.log(
    {
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "c": c,
        "im_sz": im_sz,
        "n_interval": n_interval,
        "n_updates": n_updates,
        "n_blocks": n_blocks,
        "logsnr_min": logsnr_min,
        "logsnr_max": logsnr_max,
        "latent_dim": latent_dim,
        "Langevin_clip": Langevin_clip,
        "log_path": log_path,
        "ckpt_idx": ckpt_idx,
        "load_path": load_path,
        "pi_lr": pi_lr,
        "warmup_steps": warmup_steps,
        "iterations": iterations,
        "pi_ema_decay": pi_ema_decay,
        "grad_clip": grad_clip,
        "add_q": add_q,
        "with_sn": with_sn,
        "sn_step": sn_step,
        "ebm_act": ebm_act,
        "pred_var_type": pred_var_type,
        "max_pretrain_pi_iter": max_pretrain_pi_iter,
        "print_iter": print_iter,
        "plot_iter": plot_iter,
        "ckpt_iter": ckpt_iter,
        "fid_iter": fid_iter,
        "n_fid_samples": n_fid_samples,
        "n_fid_samples_full": n_fid_samples_full,
        "reduce_variance": reduce_variance,
        "eta": eta,
    }
)

if tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp


#################### define a customized utils function #######################
class log1mexp(torch.autograd.Function):
    # From James Townsend's PixelCNN++ code
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.where(
            input > np.log(2.0),
            torch.log1p(-torch.exp(-input)),
            torch.log(-torch.expm1(-input)),
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input,) = ctx.saved_tensors
        return grad_output / torch.expm1(input)


mylog1mexp = log1mexp.apply


class tracker:
    def __init__(self, interval):
        self.record = []
        self.interval = interval

    def update(self, val):
        self.record.append(val)

    def plot(self, name):
        plt.figure()
        plt.plot(np.arange(len(self.record)) * self.interval, np.array(self.record))
        plt.savefig(name)
        plt.close()


def pred_x_from_eps(z, eps, logsnr):
    return torch.sqrt(1.0 + torch.exp(-logsnr)) * (
        z - eps * torch.rsqrt(1.0 + torch.exp(logsnr))
    )


def logsnr_schedule_fn(t, logsnr_min, logsnr_max):
    # -2log(tan(b)) == logsnr_max => b == arctan(exp(-0.5*logsnr_max))
    # -2log(tan(pi/2*a + b)) == logsnr_min
    #     => a == (arctan(exp(-0.5*logsnr_min))-b)*2/pi
    logsnr_min_tensor = logsnr_min * torch.ones_like(t)
    logsnr_max_tensor = logsnr_max * torch.ones_like(t)
    b = torch.atan(torch.exp(-0.5 * logsnr_max_tensor))
    a = torch.atan(torch.exp(-0.5 * logsnr_min_tensor)) - b
    # print(a[0], b[0], torch.exp(-0.5 * logsnr_max_tensor[0]), torch.exp(-0.5 * logsnr_min_tensor[0]))
    return -2.0 * torch.log(torch.tan(a * t + b))


def diffusion_forward(x, logsnr):
    return {
        "mean": x * torch.sqrt(F.sigmoid(logsnr)),
        "std": torch.sqrt(F.sigmoid(-logsnr)),
        "var": F.sigmoid(-logsnr),
        "logvar": torch.log(F.sigmoid(-logsnr)),
    }


def gen_samples(dev, bs, pi, xt=None):
    if xt is None:
        xt = torch.randn(bs, c, im_sz, im_sz).to(dev)
    else:
        assert False
    mystring = "step max/min: {} {:.2f}/{:.2f}".format(
        n_interval, xt.max().item(), xt.min().item()
    )
    # xts = []
    for i in reversed(range(1, n_interval + 1)):
        i_tensor = torch.ones(bs, dtype=torch.float).to(dev) * float(i)
        is_final = (i_tensor == n_interval).type(torch.float)
        logsnr_t = logsnr_schedule_fn(
            i_tensor / n_interval, logsnr_min=logsnr_min, logsnr_max=logsnr_max
        )
        logsnr_s = logsnr_schedule_fn(
            torch.clamp(i_tensor - 1.0, min=0.0) / n_interval,
            logsnr_min=logsnr_min,
            logsnr_max=logsnr_max,
        )
        xtminus1_neg0 = xt.detach().clone()
        xtminus1_neg0.requires_grad = True
        xt = Langevin(pi, xtminus1_neg0, xt, logsnr_t, logsnr_s, is_final)
        mystring += " {} {:.2f}/{:.2f}".format(i - 1, xt.max().item(), xt.min().item())
    print(mystring)
    return xt


def calculate_fid(
    dev, n_samples, pi, real_m, real_s, save_name=None, return_samples=False
):
    print("calculate fid")
    start_time = time.time()
    bs = 500
    fid_samples = []

    for i in range(n_samples // bs):
        cur_samples = gen_samples(dev, bs, pi)
        fid_samples.append(cur_samples.detach().clone())
        print(
            "Generate {} samples, time {:.2f}".format(
                (i + 1) * bs, time.time() - start_time
            )
        )

    fid_samples = torch.cat(fid_samples, dim=0)
    fid_samples = (1.0 + torch.clamp(fid_samples, min=-1.0, max=1.0)) / 2.0
    fid = pfw.fid(fid_samples, real_m=real_m, real_s=real_s, device=dev)
    if save_name is not None:
        save_images = fid_samples[:100].clone().detach().cpu()
        torchvision.utils.save_image(save_images, save_name, normalize=True, nrow=10)
    if not return_samples:
        return fid
    else:
        return fid, fid_samples


def diffusion_reverse(x, z_t, logsnr_s, logsnr_t):
    alpha_st = torch.sqrt((1.0 + torch.exp(-logsnr_t)) / (1.0 + torch.exp(-logsnr_s)))
    alpha_s = torch.sqrt(F.sigmoid(logsnr_s))
    r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
    one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)
    log_one_minus_r = mylog1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))
    mean = r * alpha_st * z_t + one_minus_r * alpha_s * x
    if pred_var_type == "large":
        var = one_minus_r * F.sigmoid(-logsnr_t)
        logvar = log_one_minus_r + torch.log(F.sigmoid(-logsnr_t))
    elif pred_var_type == "small":
        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_s)
        beta_t = 1 - a_t / a_tminus1
        var = (1.0 - a_tminus1) / (1.0 - a_t) * beta_t
        logvar = torch.log(var)
    else:
        raise NotImplemented
    return {"mean": mean, "std": torch.sqrt(var), "var": var, "logvar": logvar}


def denoise_true(z, x0, logsnr_t, logsnr_tminus1):
    z_tminus1_dist = diffusion_reverse(
        x=x0,
        z_t=z,
        logsnr_s=logsnr_tminus1.reshape(len(z), 1, 1, 1),
        logsnr_t=logsnr_t.reshape(len(z), 1, 1, 1),
    )
    a_t = F.sigmoid(logsnr_t)
    a_tminus1 = F.sigmoid(logsnr_tminus1)
    beta_t = 1 - a_t / a_tminus1
    std = torch.sqrt((1.0 - a_tminus1) / (1.0 - a_t) * beta_t).reshape(
        (len(z), 1, 1, 1)
    )
    sample_x = z_tminus1_dist["mean"] + std * torch.randn_like(z)

    if torch.any(torch.isinf(sample_x)):
        print("logsnr_tminus1", logsnr_tminus1)
        print("logsnr_t", logsnr_t)
        print("a_t", a_t)
        print("a_tminus1", a_tminus1)
        print("beta_t", beta_t)
        print("std", std)
        if torch.any(torch.isinf(z_tminus1_dist["mean"])):
            print("inf in z_tmins1_dist")

        assert False
    return sample_x


def Langevin(pi, x, xt, logsnr_t, logsnr_tminus1, is_final):
    a_t = F.sigmoid(logsnr_t)  # accumulated alpha^2 at t step
    a_tminus1 = F.sigmoid(logsnr_tminus1)  # accumulated alpha^2 at t-1 step
    as_t = (a_t / a_tminus1).reshape((len(x), 1, 1, 1))  # alpha^2 at t step
    as_t_mul = as_t.detach().clone()  # alpha^2 at t step
    as_t_mul[is_final > 0.0] = 1.0
    sigma = torch.sqrt(1.0 - as_t)  # sigma at t step

    sigma_cum = torch.sqrt(1.0 - a_tminus1)

    coeff = 1.0 / (1.0 + np.exp(-logsnr_max))

    ct_square = sigma_cum / np.sqrt(1.0 - coeff)
    sz_square = (2e-4 * ct_square).reshape((len(x), 1, 1, 1)) * sigma**2

    for _ in range(n_updates):
        en = pi(x, logsnr_tminus1)
        en = en - torch.sum((x - xt) ** 2, dim=[1, 2, 3]) * ct_square * 2e-4 * (
            1.0 - is_final
        )

        grad = torch.autograd.grad(en.sum(), [x])[0]
        x.data = x.data + 0.5 * grad + torch.sqrt(2 * sz_square) * torch.randn_like(x)

    x.data = x.data / (torch.sqrt(as_t_mul) + 1e-8)
    if Langevin_clip:
        x.data = torch.clamp(x.data, min=-1.0, max=1.0)
    x.requires_grad = False
    return x.detach()


########################## training loop ######################################


def train(index):
    if tpu:
        dev = xm.xla_device()
    else:
        dev = "cuda:0"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not tpu:
        torch.cuda.manual_seed_all(seed)

    timestamp = str(dt.datetime.now())[:19]
    timestamp = re.sub(r"[\:-]", "", timestamp)  # replace unwanted chars
    timestamp = re.sub(r"[\s]", "_", timestamp)  # with regex and re.sub

    img_dir = os.path.join(log_path, timestamp, "imgs")
    ckpt_dir = os.path.join(log_path, timestamp, "ckpt")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # shutil.copyfile(__file__, os.path.join(log_path, timestamp, __file__))

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # define dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    if tpu:
        trainloader = pl.MpDeviceLoader(trainloader, dev)
    train_iter = iter(trainloader)

    start_time = time.time()
    print("Begin calculating real image statistics")
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_test
    )
    testloader = data.DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True
    )

    fid_data_true = []
    for x, _ in testloader:
        fid_data_true.append(x)
        if len(fid_data_true) >= n_fid_samples_full:
            break
    fid_data_true = torch.cat(fid_data_true, dim=0)
    fid_data_true = (fid_data_true + 1.0) / 2.0
    real_m, real_s = pfw.get_stats(fid_data_true, device=dev)
    print(
        "Finish calculating real image statistics {:.3f}".format(
            time.time() - start_time
        ),
        fid_data_true.shape,
        fid_data_true.min(),
        fid_data_true.max(),
    )

    if training:
        fid_data_true, testset, testloader = None, None, None

    start_time = time.time()

    # begin training the model
    pi = EBM(add_q=add_q, temb_dim=128, n_blocks=n_blocks, use_sn=with_sn)
    wandb.watch(pi)
    pi.to(dev)
    pi.train()

    pi_ema = ExponentialMovingAverage(pi.parameters(), decay=pi_ema_decay)
    pi_optimizer = optim.Adam(pi.parameters(), lr=pi_lr, betas=(0.9, 0.999))
    pi_lr_scheduler = optim.lr_scheduler.LambdaLR(
        pi_optimizer,
        lr_lambda=lambda x: min(1.0, x / float(warmup_steps)),
        last_epoch=-1,
    )
    fid, fid_best = 10000, 10000

    start_iter = 0
    if load_path is not None:
        print("loading from ", load_path)
        state_dict = torch.load(load_path)
        pi.load_state_dict(state_dict["pi_state_dict"])
        pi_ema.load_state_dict(state_dict["pi_ema_state_dict"])
        pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        pi_lr_scheduler.load_state_dict(state_dict["pi_lr_scheduler"])
        start_iter = ckpt_idx

    if not training:
        pi.eval()
        true_sample_dir = os.path.join(log_path, timestamp, "true_sample")
        gen_sample_dir = os.path.join(log_path, timestamp, "gen_sample")
        os.makedirs(true_sample_dir, exist_ok=True)
        os.makedirs(gen_sample_dir, exist_ok=True)
        with pi_ema.average_parameters():
            out_fid, fid_samples = calculate_fid(
                dev,
                n_fid_samples,
                pi,
                real_m,
                real_s,
                save_name="{}/fid_samples.png".format(
                    os.path.join(log_path, timestamp)
                ),
                return_samples=True,
            )
        for i in range(len(fid_data_true)):
            torchvision.utils.save_image(
                fid_data_true[i],
                os.path.join(true_sample_dir, "{}.png".format(i)),
                normalize=True,
            )
        for i in range(len(fid_samples)):
            torchvision.utils.save_image(
                fid_samples[i],
                os.path.join(gen_sample_dir, "{}.png".format(i)),
                normalize=True,
            )
        print(out_fid)
        return

    for counter in range(start_iter, iterations):
        pi_optimizer.zero_grad()

        # generate samples from pi and p
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            x, _ = next(train_iter)

        x = x.to(dev)
        # t = torch.rand(len(x)).to(x.device) * (n_interval - 1.) / n_interval + 1.0 / n_interval # sample from [1.0 / n_interval, (n_interval-1) / n_interval)
        # t[-1] = 1.0 / n_interval
        t = (
            torch.randint(
                low=1, high=n_interval + 1, size=(len(x),), dtype=torch.float
            ).to(x.device)
            / n_interval
        )
        is_final = (t == 1.0).type(torch.float)
        t_minus1 = t - 1.0 / n_interval
        assert torch.all(t_minus1 >= 0.0)
        logsnr_t = logsnr_schedule_fn(t, logsnr_max=logsnr_max, logsnr_min=logsnr_min)
        logsnr_tminus1 = logsnr_schedule_fn(
            t_minus1, logsnr_max=logsnr_max, logsnr_min=logsnr_min
        )

        if reduce_variance:
            zt_dist = diffusion_forward(x, logsnr=logsnr_t.reshape(len(x), 1, 1, 1))
            ztminus1_dist = diffusion_forward(
                x, logsnr=logsnr_tminus1.reshape(len(x), 1, 1, 1)
            )
            tmpe = torch.randn_like(x)
            xt = zt_dist["mean"] + zt_dist["std"] * tmpe
            xtminus1 = ztminus1_dist["mean"] + ztminus1_dist["std"] * tmpe
        else:
            zt_dist = diffusion_forward(x, logsnr=logsnr_t.reshape(len(x), 1, 1, 1))
            xt = zt_dist["mean"] + zt_dist["std"] * torch.randn_like(x)
            xtminus1 = denoise_true(xt, x, logsnr_t, logsnr_tminus1)

        # calculate true energy
        pi.train()
        a_t = F.sigmoid(logsnr_t)
        a_tminus1 = F.sigmoid(logsnr_tminus1)

        a_1 = F.sigmoid(
            logsnr_schedule_fn(
                torch.tensor([1.0 / n_interval]),
                logsnr_max=logsnr_max,
                logsnr_min=logsnr_min,
            )
        )
        a_0 = F.sigmoid(
            logsnr_schedule_fn(
                torch.tensor([0.0]), logsnr_max=logsnr_max, logsnr_min=logsnr_min
            )
        )
        sigma_1 = torch.sqrt(1.0 - a_1 / a_0).item()
        as_t = a_t / a_tminus1
        beta_t = 1 - as_t
        sigma_t = torch.sqrt(beta_t)
        as_t_mul = as_t.detach().clone()
        as_t_mul[is_final > 0.0] = 1.0

        # update pi
        pi_loss_weight = 1.0 / (sigma_t / sigma_1)

        pos_energy = pi(
            xtminus1 * torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))), logsnr_tminus1
        )
        pos_loss = -(pos_energy * pi_loss_weight).mean()
        pos_loss.backward()

        # ************************************************
        # This is the sampling step inside a training step.
        # ************************************************
        xtminus1_neg0 = xt.detach().clone()
        xtminus1_neg0.requires_grad = True
        pi.eval()
        xtminus1_negk = Langevin(
            pi, xtminus1_neg0, xt, logsnr_t, logsnr_tminus1, is_final
        )
        neg_energy = pi(
            xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))),
            logsnr_tminus1,
        )

        diff = torch.sum(
            (xtminus1_negk * torch.sqrt(as_t_mul.reshape((len(x), 1, 1, 1))) - xt) ** 2
        ).item()
        neg_loss = (neg_energy * pi_loss_weight).mean()
        neg_loss.backward()
        torch.nn.utils.clip_grad_norm(pi.parameters(), max_norm=grad_clip)

        if tpu:
            xm.optimizer_step(pi_optimizer)
        else:
            pi_optimizer.step()
        pi_lr_scheduler.step()
        pi_ema.update()

        wandb.log(
            {
                "pos_loss": -pos_loss.item(),
                "neg_loss": neg_loss.item(),
                "total_loss": pos_loss.item() + neg_loss.item(),
                "diff": diff,
                "pi_lr": pi_lr_scheduler.get_last_lr()[0],
            }
        )
        if counter % print_iter == 0:
            print(
                "Iter {} time{:.2f} pos en {:.2f} neg en {:.2f} pi loss {:.2f} diff {:.2f} pi lr {}".format(
                    counter,
                    time.time() - start_time,
                    -pos_loss.item(),
                    neg_loss.item(),
                    pos_loss.item() + neg_loss.item(),
                    diff,
                    pi_lr_scheduler.get_last_lr()[0],
                )
            )

        if counter % plot_iter == 0:
            pi.eval()
            # with torch.no_grad():
            samples = gen_samples(dev, bs=64, pi=pi)
            with pi_ema.average_parameters():
                ema_samples = gen_samples(dev, bs=64, pi=pi)

            save_images = samples.detach().cpu()
            torchvision.utils.save_image(
                torch.clamp(save_images, min=-1.0, max=1.0),
                "{}/{}_p_samples.png".format(img_dir, counter),
                normalize=True,
                nrow=8,
            )
            wandb.log(
                {
                    "p_samples": wandb.Image(
                        "{}/{}_p_samples.png".format(img_dir, counter)
                    )
                }
            )
            save_images = ema_samples.detach().cpu()
            torchvision.utils.save_image(
                torch.clamp(save_images, min=-1.0, max=1.0),
                "{}/{}_ema_samples.png".format(img_dir, counter),
                normalize=True,
                nrow=8,
            )
            wandb.log(
                {
                    "ema_samples": wandb.Image(
                        "{}/{}_ema_samples.png".format(img_dir, counter)
                    )
                }
            )
            save_images = xtminus1[:64].detach().cpu()
            torchvision.utils.save_image(
                torch.clamp(save_images, min=-1.0, max=1.0),
                "{}/{}_z_tminus1.png".format(img_dir, counter),
                normalize=True,
                nrow=8,
            )
            wandb.log(
                {
                    "z_tminus1": wandb.Image(
                        "{}/{}_z_tminus1.png".format(img_dir, counter)
                    )
                }
            )
            save_images = xt[:64].detach().cpu()
            torchvision.utils.save_image(
                torch.clamp(save_images, min=-1.0, max=1.0),
                "{}/{}_z_tminus1_neg0_sample.png".format(img_dir, counter),
                normalize=True,
                nrow=8,
            )
            wandb.log(
                {
                    "z_tminus1_neg0_sample": wandb.Image(
                        "{}/{}_z_tminus1_neg0_sample.png".format(img_dir, counter)
                    )
                }
            )
            save_images = xtminus1_negk[:64].detach().cpu()
            torchvision.utils.save_image(
                torch.clamp(save_images, min=-1.0, max=1.0),
                "{}/{}_z_tminus1_negk_sample.png".format(img_dir, counter),
                normalize=True,
                nrow=8,
            )
            wandb.log(
                {
                    "z_tminus1_negk_sample": wandb.Image(
                        "{}/{}_z_tminus1_negk_sample.png".format(img_dir, counter)
                    )
                }
            )

            pi.train()

        if counter > 0 and counter % fid_iter == 0:
            fid_s_time = time.time()
            pi.eval()
            with pi_ema.average_parameters():
                fid = calculate_fid(
                    dev,
                    n_fid_samples,
                    pi,
                    real_m,
                    real_s,
                    save_name="{}/fid_samples_{}.png".format(img_dir, counter),
                )
                wandb.log({"fid": fid})
                file_to_log = "{}/fid_samples_{}.png".format(img_dir, counter)
                wandb.log({"fid_samples": wandb.Image(file_to_log), "counter": counter})
            if fid < fid_best:
                fid_best = fid
                print("Saving the best")
                save_dict = {
                    "pi_state_dict": pi.state_dict(),
                    "pi_ema_state_dict": pi_ema.state_dict(),
                    "pi_optimizer": pi_optimizer.state_dict(),
                    "pi_lr_scheduler": pi_lr_scheduler.state_dict(),
                }

                torch.save(save_dict, os.path.join(ckpt_dir, "best.pth.tar"))
            print(
                "Finish calculating fid time {:.3f} fid {:.3f} / {:.3f}".format(
                    time.time() - fid_s_time, fid, fid_best
                )
            )
            pi.train()

        if counter > 0 and (counter % ckpt_iter == 0):
            print("Saving checkpoint")
            save_dict = {
                "pi_state_dict": pi.state_dict(),
                "pi_ema_state_dict": pi_ema.state_dict(),
                "pi_optimizer": pi_optimizer.state_dict(),
                "pi_lr_scheduler": pi_lr_scheduler.state_dict(),
            }

            torch.save(save_dict, os.path.join(ckpt_dir, "{}.pth.tar".format(counter)))


if __name__ == "__main__":
    if tpu:
        xmp.spawn(train, args=())
    else:
        train(0)
