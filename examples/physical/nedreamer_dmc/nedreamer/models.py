import copy
import torch
from torch import nn
import networks
from tools import to_f32


class WorldModel(nn.Module):
    def __init__(self, config, obs_space, act_dim):
        super(WorldModel, self).__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.dynamics = networks.RSSM(config.rssm, self.embed_size, act_dim,)
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(config.decoder, self.dynamics._deter, self.dynamics.flat_stoch, shapes)
        self.heads["reward"] = networks.MLPHead(config.reward_head, self.dynamics.feat_size)
        self.heads["cont"] = networks.MLPHead(config.cont_head, self.dynamics.feat_size)

    def video_pred(self, data, initial):
        B = min(data["action"].shape[0], 6)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.dynamics.observe(
            embed[:B, :5], data["action"][:B, :5], tuple(val[:B] for val in initial), data["is_first"][:B, :5]
        )
        recon = self.heads["decoder"](post_stoch, post_deter)["image"].mode()[:B]
        # reward_post = self.heads["reward"](self.dynamics.get_feat(post_stoch, post_deter)).pred()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.dynamics.imagine_with_action(init_stoch, init_deter, data["action"][:B, 5:])
        openl = self.heads["decoder"](prior_stoch, prior_deter)["image"].mode()
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior_stoch, prior_deter)).pred()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ActorCritic(nn.Module):
    def __init__(self, config, feat_size):
        super(ActorCritic, self).__init__()
        self.actor = networks.MLPHead(config.actor, feat_size)
        self.value = networks.MLPHead(config.critic, feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._updates = 0

    def _update_slow_target(self):
        if self._updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value is always in eval mode
        self._slow_value.train(False)
        return self


class ReturnEMA(nn.Module):
    """running mean and std"""
    def __init__(self, device, alpha=1e-2):
        super(ReturnEMA, self).__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer('ema_vals', torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        flat_x = torch.flatten(to_f32(x.detach()))
        sorted_x = torch.sort(flat_x).values
        n = sorted_x.shape[0]
        indices = torch.clamp(torch.round(self.range * (n - 1)), 0, n - 1).long()
        x_quantile = sorted_x.gather(0, indices)
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
