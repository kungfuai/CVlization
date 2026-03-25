import math
import numpy as np
import re
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from tools import rpad
import distributions as dists


def cal_fan(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        init_type = "trunc_normal"
    elif isinstance(m, BlockLinear):
        shape = m.shape
        if len(shape) == 0:
            in_num, out_num = (1, 1)
        elif len(shape) == 1:
            in_num, out_num = (1, shape[0])
        elif len(shape) == 2:
            in_num, out_num = shape
        else:
            space = math.prod(shape[:-2])
            in_num, out_num = (shape[-2] * space, shape[-1] * space)
        init_type = "trunc_normal"
    elif isinstance(m, nn.RMSNorm):
        in_num, out_num = None, None
        init_type = "ones"
    elif isinstance(m, nn.Conv2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        init_type = "trunc_normal"
    elif isinstance(m, nn.Conv1d):
        space = m.kernel_size[0] * m.kernel_size[0]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        init_type = "trunc_normal"
    else:
        in_num, out_num, init_type = None, None, None
    return in_num, out_num, init_type

def weight_init_(m, fan_type="in", scale=1.0):
    in_num, out_num, init_type = cal_fan(m)
    if scale == 0.0:
        m.weight.data.fill_(0.0)
    elif init_type == "trunc_normal":
        fan = {"avg": (in_num + out_num)/2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan) * scale
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
    elif init_type == "ones":
        m.weight.data.fill_(1.0 * scale)

    if hasattr(m, "bias") and hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)


class RSSM(nn.Module):
    def __init__(
        self,
        config,
        embed_size,
        act_dim
    ):
        super(RSSM, self).__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = Deter(self._deter, self.flat_stoch, act_dim, self._hidden, blocks=self._blocks, dynlayers=self._dyn_layers, act=config.act)

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module(f"obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(f"obs_net_lambda", LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)))

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}",nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module(f"img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(f"img_net_lambda", LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)))
        self.apply(weight_init_)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1), torch.stack(logits, dim=1)

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action)

        deter = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit = self._obs_net(x)
        # ".mode()" is another option.
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        deter = self._deter_net(stoch, deter, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)

    def get_feat(self, stoch, deter):
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        config,
        shapes,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        self.selectors = []
        self.encoders = []
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.selectors.append(lambda obs: torch.cat([obs[k] for k in self.cnn_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.selectors.append(lambda obs: torch.cat([obs[k] for k in self.mlp_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)

        if len(self.encoders) > 1:
            self.fuser = lambda x: torch.cat(x, dim=-1)
        elif len(self.encoders) == 1:
            self.fuser = lambda x: x[0]
        else:
            raise NotImplementedError

        self.apply(weight_init_)

    def forward(self, obs):
        return self.fuser([enc(sel(obs)) for enc, sel in zip(self.encoders, self.selectors)])

class MultiDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        self.all_keys = list(self.mlp_shapes.keys()) + list(self.cnn_shapes.keys())

        # Unlike the encoder, each decoder is initialized independently.
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(config.cnn, deter, flat_stoch, shape,)
            self._image_dist = partial(getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist)
        if self.mlp_shapes:
            shape = (sum(sum(x) for x in self.mlp_shapes.values()),)
            config.mlp.shape = shape
            self._mlp = MLPHead(config.mlp, deter + flat_stoch)
            self._mlp_dist = partial(getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist)

    def forward(self, stoch, deter):
        dists = {}
        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            outputs = self._mlp(torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1))
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._mlp_dist(output)
                    for key, output in zip(self.mlp_shapes.keys(), outputs)
                }
            )
        return dists

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        # 1. Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding='same',
            groups=in_channels, # Depthwise Convolution
            bias=bias
        )
        # 2. Pointwise Convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for i, depth in enumerate(self.depths):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        obs = obs - 0.5
        # (batch, time, ch, h, w) -> (batch * time, ch, h, w)
        x = obs.reshape(-1, *obs.shape[-3:])
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape(x.shape[0], -1)
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shape=(3, 64, 64)):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, config.act)
        self._shape = shape
        self.depths = tuple(int(config.depth) * int(mult) for mult in list(config.mults))
        factor = 2 ** (len(self.depths))
        minres = [int(x // factor) for x in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)
        u, g = math.prod(self.min_shape), self.bspace
        self.sp0 = BlockLinear(deter, u, g)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units),
            nn.RMSNorm(2 * self.units, eps=1e-04, dtype=torch.float32),
            act())
        self.sp2 = nn.Linear(2 * self.units, math.prod(self.min_shape))
        self.sp_norm = nn.Sequential(nn.RMSNorm(self.depths[-1], eps=1e-04, dtype=torch.float32), act())
        layers = []
        in_dim = self.depths[-1]
        for i, depth in reversed(list(enumerate(self.depths[:-1]))):
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(
                Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True)
            )
            layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.float32))
            layers.append(act())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch, deter):
        B = deter.shape[:-1]
        x0, x1 = deter.reshape(B.numel(), deter.shape[-1]), stoch.reshape(B.numel(), -1)
        # (B, 1024)
        x0 = self.sp0(x0)
        # (B, g, h, w, c)
        x0 = x0.reshape(-1, self.bspace, self.min_shape[0], self.min_shape[1],  self.min_shape[2] // self.bspace)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, self.min_shape[0], self.min_shape[1], self.min_shape[2])
        # (B, 512)
        x1 = self.sp1(x1)
        # (B, 1024)
        x1 = self.sp2(x1).reshape(-1, self.min_shape[0], self.min_shape[1], self.min_shape[2])
        x = self.sp_norm(x0 + x1)
        # ch first
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # ch last
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        x = x.reshape(*B, *x.shape[1:])
        return x


class MLP(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
    ):
        super(MLP, self).__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f"{config.name}_linear{i}", nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(f"{config.name}_norm{i}", nn.RMSNorm(config.units, eps=1e-04, dtype=torch.float32))
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        if self._symlog_inputs:
            x = dists.symlog(x)
        return self.layers(x)


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super(Deter, self).__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act()
            )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act()
            )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act()
            )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        B = action.shape[0]
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)
        x = torch.cat([x0, x1, x2], -1).unsqueeze(-2).expand(-1, self.blocks, -1)
        # (B, d), (B, g, 3*h) -> (B, g, d/g + 3*h)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        # -> (B, 3*deter)
        x = self._dyn_gru(x)
        # -> (B, block, 3*deter/block)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        # -> (B, deter)
        reset, cand, update = [self.group2flat(x) for x in gates]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter

class BlockLinear(nn.Module):
    def __init__(self, in_ch, out_ch, blocks, outscale=1.0):
        super(BlockLinear, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.blocks = blocks
        self.outscale = outscale
        self.shape = (self.blocks, self.in_ch // self.blocks, self.out_ch // self.blocks)
        self.weight = nn.Parameter(torch.empty(self.shape))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        # block-wise multiplication
        x = torch.einsum('...ki,kio->...ko', x, self.weight)
        # reshape result to (..., out_ch)
        x = x.reshape(*batch_shape, self.out_ch)
        x = x + self.bias
        return x


class Conv2dSamePad(nn.Conv2d):
    def _calc_same_pad(self, i, k, s, d):
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2
                ],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    def __init__(self, ch, eps=1e-03, dtype=None):
        super(RMSNorm2D, self).__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MLPHead(nn.Module):
    def __init__(self, config, inp_dim):
        super(MLPHead, self).__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {"min_std": float(config.dist.min_std), "max_std": float(config.dist.max_std)}
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim, sum(config.shape), bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio), "shape": tuple(config.shape)}
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"device": torch.device(config.device), "bin_num": int(config.dist.bin_num)}
        elif self._dist_name == "binary":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        elif self._dist_name == "identity":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(partial(weight_init_, scale=self._outscale))

    def forward(self, x):
        return self._dist(self.last(self.mlp(x)))

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class Projector(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class NEDreamerTransformer(nn.Module):
    """Causal Transformer for NE-Dreamer: predicts encoder embeddings from RSSM feat.
    
    Takes a sequence of feat (RSSM features) and optionally actions, predicts
    encoder embeddings using causal attention with configurable output heads:
    - head_same: predict embed[t] from feat[t] (same-timestep grounding)
    - head_next_k: predict embed[t+k] from feat[t] (multi-token prediction, k = 1..predict_horizon)
    
    Architecture:
    - With actions: Interleaved [f0, a0, f1, a1, ...] tokens
      - Same-timestep: predict at feat token positions
      - Next-timestep: predict at action token positions
    - Without actions: [f0, f1, f2, ...] tokens
      - Same-timestep: predict at each position for same embed
      - Next-timestep: predict at each position for next embed
    - Causal masking ensures prediction at time t only sees up to time t
    - Multi-token prediction: separate head for each horizon k
    """
    def __init__(self, feat_dim, output_dim, action_dim, hidden_dim=256, num_layers=2, num_heads=4, 
                 max_seq_len=128, dropout=0.0, use_actions=True, act_discrete=False, act_classes=None,
                 use_same=True, use_next=True, predict_horizon=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_actions = use_actions
        self.act_discrete = act_discrete
        self.use_same = use_same
        self.use_next = use_next
        self.predict_horizon = predict_horizon
        
        assert use_same or use_next, "At least one of use_same or use_next must be True"
        assert predict_horizon >= 1, "predict_horizon must be at least 1"
        
        # Token embeddings for feat
        self.f_embed = nn.Linear(feat_dim, hidden_dim)
        
        # Action embeddings (only if use_actions=True)
        self.use_embedding = False  # Track if using nn.Embedding vs nn.Linear
        if use_actions:
            if act_discrete and act_classes is not None:
                self.a_embed = nn.Embedding(act_classes, hidden_dim)
                self.use_embedding = True
            else:
                self.a_embed = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            # Positional embeddings for interleaved sequence (2 * max_seq_len)
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_seq_len, hidden_dim))
        else:
            self.a_embed = None
            # Positional embeddings for feat-only sequence (max_seq_len)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for next-timestep prediction: predict embed[t+k] from h[t]
        # Multi-token prediction: separate head for each horizon k = 1, 2, ..., predict_horizon
        if use_next:
            self.heads_next = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(predict_horizon)
            ])
        else:
            self.heads_next = None
        
        # Output head for same-timestep prediction: predict embed[t] from h[t]
        if use_same:
            self.head_same = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head_same = None
        
        self.apply(weight_init_)
        # Re-init pos_embed after weight_init_
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, feat, actions=None):
        """
        Args:
            feat: RSSM features (B, T, feat_dim) - features from f[0] to f[T-1]
            actions: Actions (B, T, A) or (B, T) if discrete - actions from a[0] to a[T-1]
                     Only used if use_actions=True
        
        Returns:
            If dual_head=True:
                e_hat_same: Predicted same-timestep embeddings (B, T, output_dim) - predictions for e[0..T-1]
                e_hat_next: Predicted next-timestep embeddings (B, T-1, output_dim) - predictions for e[1..T-1]
            If dual_head=False:
                e_hat_next: Predicted next-timestep embeddings (B, T-1, output_dim) - predictions for e[1..T-1]
        """
        B, T, _ = feat.shape
        device = feat.device
        
        # Embed feat tokens
        tok_f = self.f_embed(feat)  # (B, T, H)
        
        if self.use_actions:
            # With actions: interleave [f0, a0, f1, a1, ...]
            assert actions is not None, "Actions required when use_actions=True"
            
            if self.use_embedding:
                # Using nn.Embedding: need integer indices
                if actions.dtype in (torch.float16, torch.float32, torch.float64):
                    # One-hot encoded: convert to indices
                    action_indices = actions.argmax(dim=-1).long()
                else:
                    # Already integer indices
                    action_indices = actions.long()
                tok_a = self.a_embed(action_indices)  # (B, T, H)
            else:
                # Using nn.Linear: need float tensors (one-hot or continuous)
                tok_a = self.a_embed(actions.float())  # (B, T, H)
            
            # Interleave: [f0, a0, f1, a1, ..., f_{T-1}, a_{T-1}]
            # Shape: (B, 2*T, H)
            tokens = torch.stack([tok_f, tok_a], dim=2).reshape(B, 2 * T, -1)
            
            # Add positional embeddings
            tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
            
            # Causal mask
            causal_mask = self._generate_causal_mask(tokens.size(1), device)
            
            # Transformer forward
            h = self.transformer(tokens, mask=causal_mask)  # (B, 2*T, H)
            
            # Extract hidden states at feat token positions for same-timestep prediction
            # f[t] is at index 2*t in interleaved sequence
            # For t=0..T-1, indices are: 0, 2, 4, ..., 2*(T-1)
            f_token_indices = torch.arange(0, 2 * T, 2, device=device)  # (T,)
            h_same = h[:, f_token_indices, :]  # (B, T, H)
            
            # Extract hidden states at action token positions for next-timestep prediction
            # To predict e[t+1], we use the hidden state after seeing a[t]
            # a[t] is at index 2*t+1 in interleaved sequence
            # For predicting e[1..T-1] from a[0..T-2], indices are: 1, 3, 5, ..., 2*(T-1)-1
            a_token_indices = torch.arange(1, 2 * T - 1, 2, device=device)  # (T-1,)
            h_next = h[:, a_token_indices, :]  # (B, T-1, H)
        else:
            # Without actions: just [f0, f1, f2, ...]
            tokens = tok_f  # (B, T, H)
            
            # Add positional embeddings
            tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
            
            # Causal mask
            causal_mask = self._generate_causal_mask(tokens.size(1), device)
            
            # Transformer forward
            h = self.transformer(tokens, mask=causal_mask)  # (B, T, H)
            
            # For same-timestep: h[t] predicts e[t]
            h_same = h  # (B, T, H)
            
            # For next-timestep: h[t] predicts e[t+1], so use h[0..T-2] to predict e[1..T-1]
            h_next = h[:, :-1, :]  # (B, T-1, H)
        
        # Project to predicted embeddings based on enabled heads
        e_hat_same = None
        e_hat_next_list = None
        
        if self.use_same:
            e_hat_same = self.head_same(h_same)  # (B, T, output_dim)
        
        if self.use_next:
            # Multi-token prediction: each head predicts a different horizon
            # heads_next[k] predicts embed[t+k+1] from h[t]
            # For horizon k, we need h[0..T-k-1] to predict embed[k+1..T]
            e_hat_next_list = []
            B, T_h, _ = h_next.shape  # h_next is (B, T-1, H) for use_actions, or needs to be computed per horizon
            
            for k in range(self.predict_horizon):
                # For horizon k+1 (1-indexed), we predict embed[t+k+1] from h[t]
                # Valid predictions: t can range from 0 to T-k-2 (so we have targets at t+k+1)
                if k < T_h:
                    h_for_k = h_next[:, :T_h - k, :]  # (B, T-1-k, H)
                    e_hat_k = self.heads_next[k](h_for_k)  # (B, T-1-k, output_dim)
                    e_hat_next_list.append(e_hat_k)
                else:
                    # Not enough sequence length for this horizon
                    e_hat_next_list.append(None)
        
        # Return based on which heads are enabled
        if self.use_same and self.use_next:
            return e_hat_same, e_hat_next_list
        elif self.use_same:
            return e_hat_same
        else:  # use_next only
            return e_hat_next_list