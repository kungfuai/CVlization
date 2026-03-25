from collections import OrderedDict
import copy
import torch
from torch import nn
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import models
import tools
from tools import to_f32
from agc import clip_grad_agc_
from laprop import LaProp
from networks import Projector, NEDreamerTransformer
import torch.nn.functional as F
import math


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super(Dreamer, self).__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = models.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.act_low = act_space.low
        self.act_high = act_space.high
        self.rep_loss = str(config.rep_loss)

        self._wm = models.WorldModel(config.world_model, obs_space, self.act_dim)
        config.behavior.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.behavior.actor.dist = config.behavior.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.behavior.actor.dist = config.behavior.actor.dist.disc
            self.act_discrete = True
        else:
            config.behavior.actor.dist = config.behavior.actor.dist.cont
        self._ac = models.ActorCritic(config.behavior, self._wm.dynamics.feat_size)
        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "dynamics": self._wm.dynamics,
            "decoder": self._wm.heads["decoder"],
            "actor": self._ac.actor,
            "value": self._ac.value,
            "reward": self._wm.heads["reward"],
            "cont": self._wm.heads["cont"],
            "encoder": self._wm.encoder,
        }
        
        if self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins alignment between projected features and encoder embeddings
            # remove decoder
            self._wm.heads.pop("decoder")
            modules.pop("decoder")
            # add projector for latent to embedding
            self.prj = Projector(self._wm.dynamics.feat_size, self._wm.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "ne_dreamer":
            # NE-Dreamer: Temporal transformer on RSSM feat → predict embeddings
            # Configurable heads:
            #   - head_same: predict embed[t] from feat[t] (same-timestep grounding)
            #   - head_next: predict embed[t+1] from feat[t] (next-timestep prediction)
            # Removes decoder, uses transformer to learn temporal structure in feat space
            self._wm.heads.pop("decoder")
            modules.pop("decoder")
            
            # Config
            cfg = config.ne_dreamer
            hidden_dim = int(cfg.hidden_dim)
            num_layers = int(cfg.num_layers)
            num_heads = int(cfg.num_heads)
            dropout = float(cfg.dropout)
            self.ne_dreamer_use_actions = bool(cfg.use_actions)
            self.ne_dreamer_use_projector = bool(cfg.use_projector)  # Project feat before transformer
            self.ne_dreamer_use_same = bool(cfg.use_same)  # Enable same-timestep head
            self.ne_dreamer_use_next = bool(cfg.use_next)  # Enable next-timestep head
            self.ne_dreamer_predict_horizon = int(cfg.predict_horizon)  # Multi-token prediction horizon
            self.ne_dreamer_horizon_discount = float(cfg.horizon_discount)  # Discount for multi-horizon
            self.ne_dreamer_loss_type = str(cfg.loss_type)  # "cosine" or "barlow"
            self.ne_dreamer_lambd = float(cfg.lambd)  # For Barlow Twins
            self.ne_dreamer_weight_same = float(cfg.weight_same)  # Weight for same-timestep loss
            self.ne_dreamer_weight_next = float(cfg.weight_next)  # Weight for next-timestep loss
            
            # Projector: feat_size -> embed_size (optional)
            if self.ne_dreamer_use_projector:
                self.prj = Projector(self._wm.dynamics.feat_size, self._wm.embed_size)
                transformer_input_dim = self._wm.embed_size
            else:
                self.prj = None
                transformer_input_dim = self._wm.dynamics.feat_size
            
            # Temporal transformer: processes feat sequence, outputs embed predictions
            self.ne_transformer = NEDreamerTransformer(
                feat_dim=transformer_input_dim,
                output_dim=self._wm.embed_size,
                action_dim=self.act_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=128,
                dropout=dropout,
                use_actions=self.ne_dreamer_use_actions,
                act_discrete=self.act_discrete,
                act_classes=act_space.n if hasattr(act_space, "n") else None,
                use_same=self.ne_dreamer_use_same,
                use_next=self.ne_dreamer_use_next,
                predict_horizon=self.ne_dreamer_predict_horizon,
            )
            
            if self.ne_dreamer_use_projector:
                modules.update({
                    "projector": self.prj,
                    "ne_transformer": self.ne_transformer,
                })
            else:
                modules.update({
                    "ne_transformer": self.ne_transformer,
                })
        elif self.rep_loss == "dreamerpro":
            # DreamerPro: Prototypical representation learning with SwAV-style loss
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._wm.heads.pop("decoder")
            modules.pop("decoder")
            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self._wm.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self._wm.dynamics.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self._wm.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        elif self.rep_loss == "dreamer":
            # DreamerV3: Standard Dreamer with reconstruction loss
            recon = self._loss_scales.pop('recon')
            self._loss_scales.update({k: recon for k in self._wm.heads["decoder"].all_keys})
        else:
            raise NotImplementedError(f"Unknown rep_loss: {self.rep_loss}. Available: r2dreamer, ne_dreamer, dreamerpro, dreamer")
            
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")
        self._agc = lambda params: clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)
        self._optimizer = LaProp(self._named_params.values(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
        self._scaler = GradScaler()
        if config.warmup:
            f = lambda step: min(1.0, (step + 1) / config.warmup)
        else:
            f = lambda step: 1.0
        self._scheduler = LambdaLR(self._optimizer, lr_lambda=f)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print('Compiling update function with torch.compile...')
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated, not whether gradients flow through its operations
        self._frozen_wm = copy.deepcopy(self._wm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._wm.named_parameters(), self._frozen_wm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_ac = copy.deepcopy(self._ac)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._ac.named_parameters(), self._frozen_ac.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self._frozen_wm.encoder(p_obs)
        prev_stoch, prev_deter, prev_action = state["stoch"], state["deter"], state["prev_action"]
        stoch, deter, _ = self._frozen_wm.dynamics.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
        feat = self._frozen_wm.dynamics.get_feat(stoch, deter)
        action_dist = self._frozen_ac.actor(feat)
        action  = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict({"stoch":stoch, "deter":deter, "prev_action":action}, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self._wm.dynamics.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch":stoch, "deter":deter, "prev_action":action}, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._wm.video_pred(p_data, initial)

    def update(self, replay_buffer):
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._ac._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer) # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None] # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values()) # clipping
        self._scaler.step(self._optimizer) # update params
        self._scaler.update() # adjust scale
        self._scheduler.step() # increment scheduler
        self._optimizer.zero_grad(set_to_none=True) # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial):
        losses = {}
        metrics = {}
        B, T = data.shape

        # World model
        embed = self._wm.encoder(data)
        rssm_input = embed
        
        post_stoch, post_deter, post_logit = self._wm.dynamics.observe(
            rssm_input, data["action"], initial, data["is_first"]
        )
        _, prior_logit  = self._wm.dynamics.prior(post_deter)
        dyn_loss, rep_loss = self._wm.dynamics.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        feat = self._wm.dynamics.get_feat(post_stoch, post_deter)
        
        # Representation learning loss based on method
        if self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins loss
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "ne_dreamer":
            # NE-Dreamer: Temporal transformer on RSSM feat → predict embeddings
            # Configurable heads:
            #   - head_same: predict embed[t] from feat[t] (same-timestep grounding)
            #   - head_next: predict embed[t+1] from feat[t] (next-timestep prediction)
            
            # Project feat to embed space (optional)
            if self.ne_dreamer_use_projector:
                transformer_input = self.prj(feat)  # (B, T, embed_size)
            else:
                transformer_input = feat  # (B, T, feat_size) - use feat directly
            
            # Prepare actions if needed
            if self.ne_dreamer_use_actions:
                a_in = data["action"]  # (B, T, A)
            else:
                a_in = None
            
            # Get predictions from transformer based on enabled heads
            # e_hat_next_list is a list of predictions for each horizon k = 1, 2, ..., predict_horizon
            if self.ne_dreamer_use_same and self.ne_dreamer_use_next:
                # Both heads enabled
                e_hat_same, e_hat_next_list = self.ne_transformer(transformer_input, a_in)
            elif self.ne_dreamer_use_same:
                # Only same-timestep head
                e_hat_same = self.ne_transformer(transformer_input, a_in)
                e_hat_next_list = None
            else:
                # Only next-timestep head
                e_hat_next_list = self.ne_transformer(transformer_input, a_in)
                e_hat_same = None
            
            # Targets - DETACHED
            e_target_same = embed.detach()  # (B, T, embed_size) - for same-timestep
            
            # Initialize losses
            loss_same = torch.tensor(0.0, device=self.device)
            loss_next = torch.tensor(0.0, device=self.device)
            
            # Compute loss based on loss_type
            off_diag_mask = ~torch.eye(self._wm.embed_size, dtype=torch.bool, device=self.device)
            
            if self.ne_dreamer_use_same and e_hat_same is not None:
                # === SAME-TIMESTEP LOSS (grounding) ===
                if self.ne_dreamer_loss_type == "cosine":
                    e_hat_same_norm = F.normalize(e_hat_same, dim=-1)
                    e_target_same_norm = F.normalize(e_target_same, dim=-1)
                    cos_sim_same = (e_hat_same_norm * e_target_same_norm).sum(dim=-1)  # (B, T)
                    loss_same = -cos_sim_same.mean()
                elif self.ne_dreamer_loss_type == "barlow":
                    N_same = B * T
                    x1_same = e_hat_same.reshape(N_same, -1)
                    x2_same = e_target_same.reshape(N_same, -1)
                    
                    x1_same_norm = (x1_same - x1_same.mean(0)) / (x1_same.std(0) + 1e-8)
                    x2_same_norm = (x2_same - x2_same.mean(0)) / (x2_same.std(0) + 1e-8)
                    
                    c_same = torch.mm(x1_same_norm.T, x2_same_norm) / N_same
                    inv_loss_same = (torch.diagonal(c_same) - 1.0).pow(2).sum()
                    red_loss_same = c_same[off_diag_mask].pow(2).sum()
                    loss_same = inv_loss_same + self.ne_dreamer_lambd * red_loss_same
            
            if self.ne_dreamer_use_next and e_hat_next_list is not None:
                # === MULTI-HORIZON NEXT-TIMESTEP LOSS (prediction) ===
                # Compute loss for each horizon k with discounting
                total_loss_next = torch.tensor(0.0, device=self.device)
                total_weight = torch.tensor(0.0, device=self.device)
                discount = self.ne_dreamer_horizon_discount
                
                for k, e_hat_k in enumerate(e_hat_next_list):
                    if e_hat_k is None:
                        continue
                    
                    # Target for horizon k+1: embed[k+1:T]
                    # e_hat_k predicts embed[t+k+1] from position t
                    # e_hat_k shape: (B, T-1-k, embed_size)
                    # Target: embed[k+1:T] -> (B, T-1-k, embed_size)
                    horizon = k + 1
                    if horizon >= T:
                        continue
                    
                    e_target_k = embed[:, horizon:, :].detach()  # (B, T-horizon, embed_size)
                    
                    # Align shapes (e_hat_k might be shorter)
                    min_len = min(e_hat_k.shape[1], e_target_k.shape[1])
                    if min_len <= 0:
                        continue
                    
                    e_hat_k = e_hat_k[:, :min_len, :]
                    e_target_k = e_target_k[:, :min_len, :]
                    
                    # Compute loss for this horizon
                    weight = discount ** k  # γ^0, γ^1, γ^2, ...
                    
                    if self.ne_dreamer_loss_type == "cosine":
                        e_hat_k_norm = F.normalize(e_hat_k, dim=-1)
                        e_target_k_norm = F.normalize(e_target_k, dim=-1)
                        cos_sim_k = (e_hat_k_norm * e_target_k_norm).sum(dim=-1)
                        loss_k = -cos_sim_k.mean()
                    elif self.ne_dreamer_loss_type == "barlow":
                        N_k = B * min_len
                        x1_k = e_hat_k.reshape(N_k, -1)
                        x2_k = e_target_k.reshape(N_k, -1)
                        
                        x1_k_norm = (x1_k - x1_k.mean(0)) / (x1_k.std(0) + 1e-8)
                        x2_k_norm = (x2_k - x2_k.mean(0)) / (x2_k.std(0) + 1e-8)
                        
                        c_k = torch.mm(x1_k_norm.T, x2_k_norm) / N_k
                        inv_loss_k = (torch.diagonal(c_k) - 1.0).pow(2).sum()
                        red_loss_k = c_k[off_diag_mask].pow(2).sum()
                        loss_k = inv_loss_k + self.ne_dreamer_lambd * red_loss_k
                    else:
                        raise ValueError(f"Unknown ne_dreamer loss_type: {self.ne_dreamer_loss_type}")
                    
                    total_loss_next = total_loss_next + weight * loss_k
                    total_weight = total_weight + weight
                
                # Normalize by total weight
                if total_weight > 0:
                    loss_next = total_loss_next / total_weight
            
            # Combined loss based on enabled heads
            if self.ne_dreamer_use_same and self.ne_dreamer_use_next:
                losses["ne_dreamer"] = (
                    self.ne_dreamer_weight_same * loss_same + 
                    self.ne_dreamer_weight_next * loss_next
                )
            elif self.ne_dreamer_use_same:
                losses["ne_dreamer"] = self.ne_dreamer_weight_same * loss_same
            else:
                losses["ne_dreamer"] = self.ne_dreamer_weight_next * loss_next
        elif self.rep_loss == "dreamerpro":
            # DreamerPro: Prototypical representation learning with SwAV-style loss
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (torch.cat([initial[0], initial[0]], dim=0), torch.cat([initial[1], initial[1]], dim=0))
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self._wm.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self._wm.dynamics.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        elif self.rep_loss == "dreamer":
            # DreamerV3: Standard Dreamer with reconstruction loss
            recon_losses = {key: torch.mean(-dist.log_prob(data[key])) for key, dist in self._wm.heads["decoder"](post_stoch, post_deter).items()}
            losses.update(recon_losses)
        else:
            raise NotImplementedError(f"Unknown rep_loss: {self.rep_loss}")

        disc = (1.0 - 1.0 / self.horizon)
        # accum rewards
        losses["rew"] = torch.mean(-self._wm.heads["reward"](feat).log_prob(to_f32(data["reward"])))
        cont = (1.0 - to_f32(data["is_terminal"]))
        losses["con"] = torch.mean(-self._wm.heads["cont"](feat).log_prob(cont))

        metrics["dyn_ent"] = torch.mean(self._wm.dynamics.get_dist(prior_logit).entropy())
        metrics["rep_ent"] = torch.mean(self._wm.dynamics.get_dist(post_logit).entropy())

        # Imagination
        start = (post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(), post_deter.reshape(-1, *post_deter.shape[2:]).detach())
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
        imag_reward = self._frozen_wm.heads["reward"](imag_feat).mode()
        # This is a probability
        imag_cont = self._frozen_wm.heads["cont"](imag_feat).mean
        imag_value = self._frozen_ac.value(imag_feat).mode()
        imag_slow_value = self._frozen_ac._slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self._ac.actor(imag_feat)
        logpi = policy.log_prob(imag_action.to(torch.bool) if self.act_discrete else imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self._ac.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(weight[:, :-1].detach() * (
            - imag_value_dist.log_prob(tar_padded.detach())
            - imag_value_dist.log_prob(imag_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_max"] = torch.max(ret_normed)
        metrics["ret_min"] = torch.min(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics['tar'] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action"] = torch.mean(imag_action)
        metrics["action_std"] = torch.std(imag_action)
        metrics["action_min"] = torch.min(imag_action)
        metrics["action_max"] = torch.max(imag_action)
        metrics["ent/action"] = torch.mean(entropy)

        # Replay
        last, term, reward = to_f32(data["is_last"]), to_f32(data["is_terminal"]), to_f32(data["reward"])
        feat = self._wm.dynamics.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_ac.value(feat).mode()
        slow_value = self._frozen_ac._slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self._ac.value(feat)
        losses['repval'] = torch.mean(weight[:, :-1] * (
            - value_dist.log_prob(ret_padded.detach())
            - value_dist.log_prob(slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        feats = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            feat = self._frozen_wm.dynamics.get_feat(stoch, deter)
            # should this be rsample?
            action = self._frozen_ac.actor(feat).rsample()
            # The feat and the action "taken from it" are stored at same index.
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_wm.dynamics.img_step(stoch, deter, action)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    # DreamerPro helper methods
    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # image shape is (B, T, H, W, C), but random_translate expects (B, T, C, H, W)
        image = data_aug['image'].permute(0, 1, 4, 2, 3)
        data_aug['image'] = self.random_translate(image, self.aug_max_delta, same_across_time=self.aug_same_across_time, bilinear=self.aug_bilinear)
        data_aug['image'] = data_aug['image'].permute(0, 1, 3, 4, 2)  # permute back
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
            proj = F.normalize(proj, p=2, dim=-1)
        return proj

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._updates > 0 else 1.0
            for s, d in zip(self._wm.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._updates += 1

    def sinkhorn(self, scores):
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up:]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up:]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)
        
        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self._wm.dynamics.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up:]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = (
            -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0))
            -0.5 * torch.mean(torch.sum(ema_targets_1 * obs_logits_2, dim=0))
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = (
            torch.mean(torch.square(obs_norm - 1))
            + torch.mean(torch.square(feat_norm - 1))
        )

        losses = {
            'swav': swav_loss,
            'temp': temp_loss,
            'norm': norm_loss,
        }
        return losses

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)
        
        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = 'bilinear' if bilinear else 'nearest'
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)
        
        return x_translated.reshape(B, T, C, H, W)
