"""
Hyper light speed bench.

Adapted from https://github.com/KellerJordan/hlb-CIFAR10
"""
#############################################
#            Setup/Hyperparameters          #
#############################################

import sys
import uuid
import numpy as np

from functools import partial
import math
import os
import copy

import torch
from torch import nn
import torch.nn.functional as F

## <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). That said, if there's
# ways this code could be improved and cleaned up, please do open a PR on the GitHub repo. Your support and help is much appreciated for this
# project! :)


# This is for testing that certain changes don't exceed some X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
# torch.cuda.set_per_process_memory_fraction(fraction=6.5/40., device=0) ## 40. GB is the maximum memory of the base A100 GPU

# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {"kernel_size": 3, "padding": "same", "bias": False}

batchsize = 1024
bias_scaler = 64
# To replicate the ~95.79%-accuracy-in-110-seconds runs, you can change the base_depth from 64->128, train_epochs from 12.1->90, ['ema'] epochs 10->80, cutmix_size 3->10, and cutmix_epochs 6->80
hyp = {
    "opt": {
        "bias_lr": 1.525 * bias_scaler / 1024,
        "non_bias_lr": 1.525 / 1024,
        "bias_decay": 2 * 6.687e-4 * batchsize / bias_scaler,
        "non_bias_decay": 2 * 6.687e-4 * batchsize,
        "scaling_factor": 1.0 / 9,
        "percent_start": 0.23,
        "loss_scale": 32,
    },
    "net": {
        "whitening": {
            "kernel_size": 2,
        },
        "pad_amount": 2,
        "cutout_size": 0,
        "batch_norm_momentum": 0.4,  # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        "base_depth": 64,  ## This should be a factor of 8 in some way to stay tensor core friendly
    },
    "misc": {
        "ema": {
            "start_epochs": 2,
            "decay_base": 0.95,
            "decay_pow": 3.0,
            "every_n_steps": 5,
        },
        "train_epochs": 11.5,
    },
}

#############################################
#                DataLoader                 #
#############################################

## https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n, c, h, w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h - size + 1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w - size + 1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(
        1, 1, h, 1
    ) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(
        1, 1, 1, w
    ) - corner_x.view(-1, 1, 1, 1)

    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(inputs, crop_size):
    crop_mask = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)


def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


## This is a pre-padded variant of quick_cifar.CifarLoader which moves the padding step of random translate
## from __iter__ to __init__, so that it doesn't need to be repeated each epoch.
class PrepadCifarLoader:
    def __init__(
        self,
        path,
        train=True,
        batch_size=500,
        aug=None,
        drop_last=None,
        shuffle=None,
        gpu=0,
    ):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.denormalize = T.Normalize(-CIFAR_MEAN / CIFAR_STD, 1 / CIFAR_STD)

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate", "cutout"], "Unrecognized key: %s" % k

        # Pre-pad images to save time when doing random translation
        pad = self.aug.get("translate", 0)
        self.prepad_images = F.pad(self.images, (pad,) * 4, "reflect")

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def augment_prepad(self, images):
        images = self.normalize(images)
        images = batch_crop(images, self.images.shape[-2])
        if self.aug.get("flip", False):
            images = batch_flip_lr(images)
        if self.aug.get("cutout", 0) > 0:
            images = batch_cutout(images, self.aug["cutout"])
        return images

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):
        images = self.augment_prepad(self.prepad_images)
        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Components             #
#############################################


# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-12,
        momentum=hyp["net"]["batch_norm_momentum"],
        weight=False,
        bias=True,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs


class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.temperature = temperature

    def forward(self, x):
        if self.temperature is not None:
            weight = self.weight * self.temperature
        else:
            weight = self.weight
        return x @ weight.T


# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x


class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2, 3))  # Global maximum pooling


#############################################
#          Init Helper Functions            #
#############################################


def get_patches(x, patch_shape):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
    c, (h, w) = x.shape[1], patch_shape
    return (
        x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    )


def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n, c, h, w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(
        c * h * w, c, h, w
    ).flip(0)


# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
# Note that this is a large epsilon, so the bottom half of principal directions won't fully whiten
def init_whitening_conv(layer, train_set, eps=1e-2):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(
        eigenvalues + eps
    )  # set the filters as the eigenvectors in order to whiten inputs
    eigenvectors_scaled_truncated = eigenvectors_scaled[: len(layer.weight) // 2]
    layer.weight.data[:] = torch.cat(
        (eigenvectors_scaled_truncated, -eigenvectors_scaled_truncated)
    )
    ## We don't want to train this, since this is implicitly whitening over the whole dataset
    ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    layer.weight.requires_grad = False


#############################################
#            Network Definition             #
#############################################

scaler = 2.0  ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    "block1": round(
        scaler**0 * hyp["net"]["base_depth"]
    ),  # 64  w/ scaler at base value
    "block2": round(
        scaler**2 * hyp["net"]["base_depth"]
    ),  # 256 w/ scaler at base value
    "block3": round(
        scaler**3 * hyp["net"]["base_depth"]
    ),  # 512 w/ scaler at base value
    "num_classes": 10,
}


class SpeedyConvNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict  # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict["initial_block"]["whiten"](x)
        x = self.net_dict["initial_block"]["activation"](x)
        x = self.net_dict["conv_group_1"](x)
        x = self.net_dict["conv_group_2"](x)
        x = self.net_dict["conv_group_3"](x)
        x = self.net_dict["pooling"](x)
        x = self.net_dict["linear"](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(len(x) // 2, dim=0)
            x = 0.5 * orig + 0.5 * flipped
        return x


def make_net(train_images):
    whiten_conv_depth = 2 * 3 * hyp["net"]["whitening"]["kernel_size"] ** 2
    network_dict = nn.ModuleDict(
        {
            "initial_block": nn.ModuleDict(
                {
                    "whiten": Conv(
                        3,
                        whiten_conv_depth,
                        kernel_size=hyp["net"]["whitening"]["kernel_size"],
                        padding=0,
                    ),
                    "activation": nn.GELU(),
                }
            ),
            "conv_group_1": ConvGroup(whiten_conv_depth, depths["block1"]),
            "conv_group_2": ConvGroup(depths["block1"], depths["block2"]),
            "conv_group_3": ConvGroup(depths["block2"], depths["block3"]),
            "pooling": FastGlobalMaxPooling(),
            "linear": Linear(
                depths["block3"],
                depths["num_classes"],
                bias=False,
                temperature=hyp["opt"]["scaling_factor"],
            ),
        }
    )

    net = SpeedyConvNet(network_dict)
    net = net.cuda()
    net = net.to(
        memory_format=torch.channels_last
    )  # to appropriately use tensor cores/avoid thrash while training
    net.train()
    net.half()  # Convert network to half before initializing the initial whitening layer.

    with torch.no_grad():
        init_whitening_conv(
            net.net_dict["initial_block"]["whiten"], train_images[:5000]
        )

        for name, block in net.net_dict.items():
            if "conv_group" in name:
                # Create an implicit residual via a dirac-initialized tensor
                dirac_weights_in = torch.nn.init.dirac_(
                    torch.empty_like(block.conv1.weight)
                )

                # Add the implicit residual to the already-initialized convolutional transition layer.
                # One can use more sophisticated initializations, but this one appeared worked best in testing.
                # What this does is brings up the features from the previous residual block virtually, so not only
                # do we have residual information flow within each block, we have a nearly direct connection from
                # the early layers of the network to the loss function.
                std_pre, mean_pre = torch.std_mean(block.conv1.weight.data)
                block.conv1.weight.data = block.conv1.weight.data + dirac_weights_in
                std_post, mean_post = torch.std_mean(block.conv1.weight.data)

                # Renormalize the weights to match the original initialization statistics
                block.conv1.weight.data.sub_(mean_post).div_(std_post).mul_(
                    std_pre
                ).add_(mean_pre)

                ## We do the same for the second layer in each convolution group block, since this only
                ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
                ## convolution. This can be easily scaled down by the network, and the weights can more easily
                ## pivot in whichever direction they need to go now.
                ## The reason that I believe that this works so well is because a combination of MaxPool2d
                ## and the nn.GeLU function's positive bias encouraging values towards the nearly-linear
                ## region of the GeLU activation function at network initialization. I am not currently
                ## sure about this, however, it will require some more investigation. For now -- it works! D:
                torch.nn.init.dirac_(block.conv2.weight)

    return net


########################################
#          Training Helpers            #
########################################


class NetworkEMA(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net_ema = copy.deepcopy(net).eval()

    def update(self, net, decay):
        with torch.no_grad():
            for net_ema_param, (param_name, net_param) in zip(
                self.net_ema.state_dict().values(), net.state_dict().items()
            ):
                if net_param.dtype in (torch.half, torch.float):
                    net_ema_param.lerp_(
                        net_param.detach(), 1 - decay
                    )  # linear interpolation
                    # And then we also copy the parameters back to the network, similarly to the Lookahead optimizer (but with a much more aggressive-at-the-end schedule)
                    # if not ('norm' in param_name and 'weight' in param_name) and not 'whiten' in param_name:
                    if not "whiten" in param_name:
                        net_param.copy_(net_ema_param.detach())

    def forward(self, inputs):
        return self.net_ema(inputs)


def init_split_parameter_dictionaries(network):
    loss_scale = hyp["opt"]["loss_scale"]
    wd_nonbias = hyp["opt"]["non_bias_decay"] * loss_scale
    wd_bias = hyp["opt"]["bias_decay"] * loss_scale
    lr_nonbias = hyp["opt"]["non_bias_lr"] / loss_scale
    lr_bias = hyp["opt"]["bias_lr"] / loss_scale
    hyp_nonbias = {
        "params": [],
        "lr": lr_nonbias,
        "momentum": 0.85,
        "nesterov": True,
        "weight_decay": wd_nonbias,
    }
    hyp_bias = {
        "params": [],
        "lr": lr_bias,
        "momentum": 0.85,
        "nesterov": True,
        "weight_decay": wd_bias,
    }
    for name, p in network.named_parameters():
        if p.requires_grad:
            if "bias" in name:
                hyp_bias["params"].append(p)
            else:
                hyp_nonbias["params"].append(p)
    return hyp_nonbias, hyp_bias


logging_columns_list = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "ema_val_acc",
    "total_time_seconds",
]


# define the printing function and print the column heads
def print_training_details(
    columns_list,
    separator_left="|  ",
    separator_right="  ",
    final="|",
    column_heads_only=False,
    is_final_entry=False,
):
    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print("-" * (len(print_string)))  # print the top bar
        print(print_string)
        print("-" * (len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print("-" * (len(print_string)))  # print the final output bar


########################################
#           Train and Eval             #
########################################


def main():
    # Initializing constants for the whole run.
    net_ema = None

    total_time_seconds = 0.0
    current_steps = 0.0

    train_augs = dict(
        flip=True, translate=hyp["net"]["pad_amount"], cutout=hyp["net"]["cutout_size"]
    )
    train_loader = PrepadCifarLoader(
        "/tmp/cifar10", train=True, batch_size=batchsize, aug=train_augs
    )
    test_loader = PrepadCifarLoader("/tmp/cifar10", train=False, batch_size=2500)

    # Get network
    train_images = train_loader.normalize(train_loader.images)
    net = make_net(train_images)

    # Loss function is smoothed cross-entropy
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction="none")

    # One optimizer for the regular network, and one for the biases.
    non_bias_params, bias_params = init_split_parameter_dictionaries(net)
    opt = torch.optim.SGD([non_bias_params, bias_params])

    # Learning rate and EMA scheduling
    total_train_steps = math.ceil(len(train_loader) * hyp["misc"]["train_epochs"])
    # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
    pct_start = hyp["opt"][
        "percent_start"
    ]  # * (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))
    final_lr_ratio = 0.07  # Actually pretty important, apparently!
    lr_schedule = np.interp(
        np.arange(1 + total_train_steps),
        [0, int(pct_start * total_train_steps), total_train_steps],
        [0, 1, final_lr_ratio],
    )
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
    ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
    ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
    projected_ema_decay_val = (
        hyp["misc"]["ema"]["decay_base"] ** hyp["misc"]["ema"]["every_n_steps"]
    )

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()  ## clean up any pre-net setup operations

    for epoch in range(math.ceil(hyp["misc"]["train_epochs"])):
        #################
        # Training Mode #
        #################

        torch.cuda.synchronize()
        starter.record()
        net.train()

        for epoch_step, (inputs, labels) in enumerate(train_loader):
            outputs = net(inputs)
            loss = loss_fn(outputs, labels).sum()
            loss_scaled = hyp["opt"]["loss_scale"] * loss
            loss_scaled.backward()

            opt.step()
            opt.zero_grad(set_to_none=True)
            lr_sched.step()

            current_steps += 1

            if (
                epoch >= hyp["misc"]["ema"]["start_epochs"]
                and current_steps % hyp["misc"]["ema"]["every_n_steps"] == 0
            ):
                ## Initialize the ema from the network at this point in time if it does not already exist.... :D
                if net_ema is None:  # don't snapshot the network yet if so!
                    net_ema = NetworkEMA(net)
                else:
                    # We warm up our ema's decay/momentum value over training exponentially according to the hyp config dictionary (this lets us move fast, then average strongly at the end).
                    net_ema.update(
                        net,
                        decay=projected_ema_decay_val
                        * (current_steps / total_train_steps)
                        ** hyp["misc"]["ema"]["decay_pow"],
                    )

            if current_steps >= total_train_steps:
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        # Evaluation  Mode #
        ####################

        ## Get the accuracy and loss of the last batch of training data
        train_acc = (outputs.detach().argmax(-1) == labels).float().mean().item()
        train_loss = loss.item() / batchsize

        net.eval()
        with torch.no_grad():
            loss_list, acc_list, acc_list_ema = [], [], []
            for inputs, labels in test_loader:
                outputs = net(inputs)
                loss_list.append(loss_fn(outputs, labels).float().mean())
                acc_list.append((outputs.argmax(-1) == labels).float().mean())
                if net_ema:
                    outputs = net_ema(inputs)
                    acc_list_ema.append((outputs.argmax(-1) == labels).float().mean())
            val_acc = torch.stack(acc_list).mean().item()
            val_loss = torch.stack(loss_list).mean().item()
            ema_val_acc = None
            if net_ema:
                ema_val_acc = torch.stack(acc_list_ema).mean().item()

        # We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
        ## Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
        ## more heinous than this, unfortunately. So we switched to the "more simple" version of this!
        format_for_table = (
            lambda x, locals: (f"{locals[x]}".rjust(len(x)))
            if type(locals[x]) == int
            else "{:0.4f}".format(locals[x]).rjust(len(x))
            if locals[x] is not None
            else " " * len(x)
        )

        # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
        ## We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
        print_training_details(
            list(map(partial(format_for_table, locals=locals()), logging_columns_list)),
            is_final_entry=(epoch >= math.ceil(hyp["misc"]["train_epochs"] - 1)),
        )

    return ema_val_acc  # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)


if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    print_training_details(
        logging_columns_list, column_heads_only=True
    )  ## print out the training column heads before we print the actual content for each run.
    acc_list = []
    for run_num in range(25):
        acc_list.append(torch.tensor(main()))
    print(
        "Mean/std:",
        (
            torch.mean(torch.stack(acc_list)).item(),
            torch.std(torch.stack(acc_list)).item(),
        ),
    )

    log = {"code": code, "accs": acc_list}
    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    print(log_path)
    torch.save(log, os.path.join(log_dir, "log.pt"))
