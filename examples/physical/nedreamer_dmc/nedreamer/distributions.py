import torch
from torch.nn import functional as F
from torch import distributions as torchd

from tools import to_f32, to_i32

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        _mode = F.one_hot(
            torch.argmax(self.logits, axis=-1), self.logits.shape[-1]
        )
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=()):
        sample = super().sample(sample_shape).detach()
        probs = self.probs
        for _ in range(len(sample.shape) - len(probs.shape)):
            probs = probs.unsqueeze(0)
        sample = sample + probs - probs.detach()
        return sample

    def sample(self, **kwargs):
        raise NotImplementedError

class MultiOneHotDist:
    def __init__(self, logits, shape, unimix_ratio=0.0):
        self.shape = shape
        splits = torch.split(logits, shape, dim=-1)
        self.onehots = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self):
        _modes = [dist.mode for dist in self.onehots]
        return torch.cat(_modes, dim=-1)

    def rsample(self, sample_shape=()):
        _rsamples = [dist.rsample() for dist in self.onehots]
        return torch.cat(_rsamples, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_prob(self, value):
        splits = torch.split(value, self.shape, dim=-1)
        _log_probs = [dist.log_prob(s) for dist, s in zip(self.onehots, splits)]
        return sum(_log_probs)

    def entropy(self):
        _entropies = [dist.entropy() for dist in self.onehots]
        return sum(_entropies)


def kl(logits_left, logits_right):
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)

class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + (
                (p1 * b1).flip(dims=(-1,)) + (p3 * b3)
            ).sum(dim=-1, keepdim=True)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., : n // 2]
            p2 = self.probs[..., n // 2 :]
            b1 = self.bins[..., : n // 2]
            b2 = self.bins[..., n // 2 :]
            wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
            return self.unsquash(wavg)

    def log_prob(self, target):
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)
        target_squashed = self.squash(target).detach()
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = (below == above)
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs()
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs()
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        mixed_target = (
            oh_below * weight_below.unsqueeze(-1) +
            oh_above * weight_above.unsqueeze(-1)
        )
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (mixed_target * log_pred).sum(dim=-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = to_f32(mode)
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class Bound:
    def __init__(self, dist):
        super().__init__()
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        out = self._dist.mean
        out = out / torch.clip(torch.abs(out), min=1.0).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        out = out / torch.clip(torch.abs(out), min=1.0).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    # NOTE: Bound can be added
    dist = torchd.normal.Normal(torch.tanh(to_f32(mean)), to_f32(std))
    dist = torchd.independent.Independent(dist, 1)
    return dist

def normal_std_fixed(mean, std, **kwargs):
    dist = torchd.normal.Normal(to_f32(mean), to_f32(std))
    dist = Bound(torchd.independent.Independent(dist, 1))
    return dist

def onehot(mean, unimix_ratio, **kwargs):
    dist = OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)
    return dist

def multi_onehot(mean, unimix_ratio, shape, **kwargs):
    dist = MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)
    return dist

def binary(logits, **kwargs):
    dist = torchd.independent.Independent(
            torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1
        )
    return dist

def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    dist = TwoHot(to_f32(logits), bins)
    return dist

def symlog_mse(logits, **kwargs):
    dist = SymlogDist(to_f32(logits))
    return dist

def mse(logits, **kwargs):
    return MSEDist(to_f32(logits))

def identity(logits, **kwargs):
    return logits

def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	"""Sample from the Gumbel-Softmax distribution."""
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)
