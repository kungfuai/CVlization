from pathlib import Path
import os
import sys
import shutil
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
import wandb

from torchvision.utils import save_image, make_grid

# from tensorboardX import SummaryWriter
from . import cfg
from .models import Res12_Quadratic, Res18_Quadratic, Res34_Quadratic
from .models import Res12_Quadratic, Res18_Quadratic
from .sampling import (
    Langevin_E,
    SS_denoise,
    Annealed_Langevin_E,
    Reverse_AIS_sampling,
    AIS_sampling,
)

# from models.SE_ResNet import SE_Res18_Quadratic, Swish
# import pdb


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.rand_seed)

    # switch datasets and models

    wandb.init(project="mdsm", config=args)

    if args.dataset == "cifar":
        from .data.cifar import inf_train_gen

        itr = inf_train_gen(args.batch_size, flip=False)
        netE = Res18_Quadratic(3, args.n_chan, 32, normalize=False, AF=nn.ELU())
        # netE = SE_Res18_Quadratic(3,args.n_chan,32,normalize=False,AF=Swish())

    elif args.dataset == "mnist":
        from .data.mnist_32 import inf_train_gen

        itr = inf_train_gen(args.batch_size)
        netE = Res12_Quadratic(1, args.n_chan, 32, normalize=False, AF=nn.ELU())

    elif args.dataset == "fmnist":
        # print(dataset+str(args.n_chan))
        from .data.fashion_mnist_32 import inf_train_gen

        itr = inf_train_gen(args.batch_size)
        netE = Res12_Quadratic(1, args.n_chan, 32, normalize=False, AF=nn.ELU())

    else:
        NotImplementedError("{} unknown dataset".format(args.dataset))

    # setup gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netE = netE.to(device)
    if args.n_gpus > 1:
        netE = nn.DataParallel(netE)

    # setup path

    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    # pdb.set_trace()
    print(str(args.cont))
    # print(str(args.time))
    if args.cont == True:
        root = "logs/" + args.log + "_" + args.time  # compose string for loading
        # load network
        file_name = "netE_" + str(args.net_indx) + ".pt"
        netE.load_state_dict(torch.load(root + "/models/" + file_name))
    else:  # start new will create logging folder
        root = "logs/" + args.log + "_" + timestamp  # add timestemp
        # over write if folder already exist, not likely to happen as timestamp is used
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.makedirs(root)
        os.makedirs(root + "/models")
        os.makedirs(root + "/samples")

    # setup optimizer and lr scheduler
    params = {"lr": args.max_lr, "betas": (0.9, 0.95)}
    optimizerE = torch.optim.Adam(netE.parameters(), **params)
    if args.lr_schedule == "exp":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerE, int(args.n_iter / 6))

    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerE, args.n_iter, eta_min=1e-6, last_epoch=-1
        )

    elif args.lr_schedule == "const":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerE, int(args.n_iter))

    # train
    print_interval = 50
    max_iter = args.n_iter + args.net_indx
    batchSize = args.batch_size
    sigma0 = 0.1
    sigma02 = sigma0**2

    if args.noise_distribution == "exp":
        sigmas_np = np.logspace(
            np.log10(args.min_noise), np.log10(args.max_noise), batchSize
        )
    elif args.noise_distribution == "lin":
        sigmas_np = np.linspace(args.min_noise, args.max_noise, batchSize)

    sigmas = torch.Tensor(sigmas_np).view((batchSize, 1, 1, 1)).to(device)

    start_time = time.time()

    for i in range(args.net_indx, args.net_indx + args.n_iter):
        x_real = itr.__next__().to(device)
        x_noisy = x_real + sigmas * torch.randn_like(x_real)

        x_noisy = x_noisy.requires_grad_()
        E = netE(x_noisy).sum()
        grad_x = torch.autograd.grad(E, x_noisy, create_graph=True)[0]
        x_noisy.detach()

        optimizerE.zero_grad()

        LS_loss = (
            (((x_real - x_noisy) / sigmas / sigma02 + grad_x / sigmas) ** 2) / batchSize
        ).sum()

        LS_loss.backward()
        optimizerE.step()
        scheduler.step()

        if (i + 1) % print_interval == 0:
            time_spent = time.time() - start_time
            start_time = time.time()
            netE.eval()
            E_real = netE(x_real).mean()
            E_noise = netE(torch.rand_like(x_real)).mean()
            netE.train()

            print(
                "Iteration {}/{} ({:.0f}%), E_real {:e}, E_noise {:e}, Normalized Loss {:e}, time {:4.1f}".format(
                    i + 1,
                    max_iter,
                    100 * ((i + 1) / max_iter),
                    E_real.item(),
                    E_noise.item(),
                    (sigma02**2) * (LS_loss.item()),
                    time_spent,
                )
            )

            # writer.add_scalar("E_real", E_real.item(), i + 1)
            # writer.add_scalar("E_noise", E_noise.item(), i + 1)
            # writer.add_scalar("loss", (sigma02**2) * LS_loss.item(), i + 1)
            wandb.log(
                {
                    "E_real": E_real.item(),
                    "E_noise": E_noise.item(),
                    "loss": (sigma02**2) * LS_loss.item(),
                }
            )
            del E_real, E_noise, x_real, x_noisy

        if (i + 1) % args.save_every == 0:
            print("-" * 50)
            file_name = args.file_name + str(i + 1) + ".pt"
            torch.save(netE.state_dict(), root + "/models/" + file_name)

        # if (i + 1) % args.save_every == 0:
        if (i) % 1000 == 0:
            # TODO: use a different arg to control sampling frequency
            print("-" * 50)
            print("Sampling...")

            if args.dataset == "cifar":
                sample_x = torch.zeros((args.batch_size, 3, 32, 32))
            else:
                raise NotImplementedError(
                    "Sampling for {} is not supported".format(args.dataset)
                )

            if args.annealing_schedule == "exp":
                Nsampling = 2000  # exponential schedule with flat region in the beginning and end
                Tmax, Tmin = 100, 1
                T = Tmax * np.exp(
                    -np.linspace(0, Nsampling - 1, Nsampling)
                    * (np.log(Tmax / Tmin) / Nsampling)
                )
                T = np.concatenate((Tmax * np.ones((500,)), T), axis=0)
                T = np.concatenate((T, Tmin * np.linspace(1, 0, 200)), axis=0)

            elif args.annealing_schedule == "lin":
                Nsampling = (
                    2000  # linear schedule with flat region in the beginning and end
                )
                Tmax, Tmin = 100, 1
                T = np.linspace(Tmax, Tmin, Nsampling)
                T = np.concatenate((Tmax * np.ones((500,)), T), axis=0)
                T = np.concatenate((T, Tmin * np.linspace(1, 0, 200)), axis=0)

            n_batches = int(np.ceil(args.n_samples_save / args.batch_size))
            denoise_samples = []
            print("sampling starts")
            n_batches = 1
            for i in range(n_batches):
                print("batch {}/{} starts".format((i + 1), n_batches))
                initial_x = 0.5 + torch.randn_like(sample_x).to(device)
                x_list, E_trace = Annealed_Langevin_E(
                    netE, initial_x, args.sample_step_size, T, 100
                )

                x_denoise = SS_denoise(x_list[-1][:].to(device), netE, 0.1)
                denoise_samples.append(x_denoise)
                print("batch {}/{} finished".format((i + 1), n_batches))

            denoise_samples = torch.cat(denoise_samples, 0)
            # denoise_samples has shape (n_samples, 3, 32, 32)

            torchvision.utils.save_image(
                denoise_samples,
                root
                + "/samples/"
                + args.dataset
                + "_"
                + str(args.n_samples_save)
                + "step"
                + str(i)
                + "_samples.png",
                nrow=8,
                normalize=True,
            )
            wandb.log(
                {
                    "samples": [
                        wandb.Image(
                            torchvision.utils.make_grid(
                                denoise_samples, nrow=8, normalize=True
                            )
                        )
                    ]
                }
            )

            # torch.save(
            #     denoise_samples,
            #     root
            #     + "/samples/"
            #     + args.dataset
            #     + "_"
            #     + str(args.n_samples_save)
            #     + "samples.pt",
            # )


if __name__ == "__main__":
    main()
