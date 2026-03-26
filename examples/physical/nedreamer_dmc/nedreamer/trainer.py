import numpy as np
import torch
import tensordict
from tensordict import TensorDict

import tools
import saliency
from posthoc_decoder import PostHocDecoder


class OnlineTrainer:
    def  __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, act_dim):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.act_dim = act_dim

        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        
        # Eval video logging settings
        self.eval_video_every = int(config.eval_video_every) if config.eval_video_every else 0
        self.s3_bucket = config.s3_bucket
        self.s3_prefix = config.s3_prefix if config.s3_prefix else "ne_dreamer"
        self._eval_count = 0  # Counter for eval video logging
        
        # Saliency map settings
        self.saliency_enabled = bool(config.saliency_enabled) if hasattr(config, 'saliency_enabled') else False
        self.saliency_stride = int(config.saliency_stride) if hasattr(config, 'saliency_stride') else 5
        self.saliency_mask_sigma = float(config.saliency_mask_sigma) if hasattr(config, 'saliency_mask_sigma') else 5.0
        self.saliency_blur_sigma = float(config.saliency_blur_sigma) if hasattr(config, 'saliency_blur_sigma') else 3.0
        self.saliency_every_n = int(config.saliency_every_n) if hasattr(config, 'saliency_every_n') else 1
        self.saliency_alpha = float(config.saliency_alpha) if hasattr(config, 'saliency_alpha') else 0.5
        self.saliency_colormap = str(config.saliency_colormap) if hasattr(config, 'saliency_colormap') else 'hot'
        
        # Post-hoc decoder settings
        self.posthoc_decoder_enabled = bool(config.posthoc_decoder_enabled) if hasattr(config, 'posthoc_decoder_enabled') else False
        self.posthoc_decoder_hidden_dim = int(config.posthoc_decoder_hidden_dim) if hasattr(config, 'posthoc_decoder_hidden_dim') else 256
        self.posthoc_decoder_depth = int(config.posthoc_decoder_depth) if hasattr(config, 'posthoc_decoder_depth') else 32
        self.posthoc_decoder_lr = float(config.posthoc_decoder_lr) if hasattr(config, 'posthoc_decoder_lr') else 1e-4
        self.posthoc_decoder_train_steps = int(config.posthoc_decoder_train_steps) if hasattr(config, 'posthoc_decoder_train_steps') else 100
        self.posthoc_decoder_imag_horizon = int(config.posthoc_decoder_imag_horizon) if hasattr(config, 'posthoc_decoder_imag_horizon') else 15
        self.posthoc_decoder_counterfactuals = bool(config.posthoc_decoder_counterfactuals) if hasattr(config, 'posthoc_decoder_counterfactuals') else False
        self.posthoc_decoder_num_cf_actions = int(config.posthoc_decoder_num_cf_actions) if hasattr(config, 'posthoc_decoder_num_cf_actions') else 3
        self._posthoc_decoder = None  # Lazy initialization
        
        # Imagination decoding settings (open-loop prediction visualization)
        self.imag_decoding_enabled = bool(config.imag_decoding_enabled) if hasattr(config, 'imag_decoding_enabled') else False
        self.imag_decoding_every_k = int(config.imag_decoding_every_k) if hasattr(config, 'imag_decoding_every_k') else 10
        self.imag_decoding_context_len = int(config.imag_decoding_context_len) if hasattr(config, 'imag_decoding_context_len') else 5
        self.imag_decoding_pred_horizon = int(config.imag_decoding_pred_horizon) if hasattr(config, 'imag_decoding_pred_horizon') else 45
        self.imag_decoding_num_samples = int(config.imag_decoding_num_samples) if hasattr(config, 'imag_decoding_num_samples') else 3
        self.imag_decoding_show_uncertainty = bool(config.imag_decoding_show_uncertainty) if hasattr(config, 'imag_decoding_show_uncertainty') else True
        self.imag_decoding_uncertainty_samples = int(config.imag_decoding_uncertainty_samples) if hasattr(config, 'imag_decoding_uncertainty_samples') else 4

    def to_td(self, transition):
        if "reward" not in transition:
            transition.update(reward=torch.zeros((1,), dtype=torch.float32))
        transition = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in transition.items()}
        td = TensorDict(transition, batch_size=())
        for key in td.keys():
            if td[key].ndim == 0 and key != "episode":
                td[key] = td[key].unsqueeze(-1)
        return td.unsqueeze(0)

    def eval(self, agent, train_step):
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        # Track success (success at any point in episode counts)
        successes = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        act = agent_state["prev_action"].clone() # (B, A)
        while not once_done.all():
            steps += ~done * ~once_done
            # Step envs on CPU to avoid GPU<->CPU sync in the worker processes
            act_cpu = act.detach().to('cpu')
            done_cpu = done.detach().to('cpu')
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent
            trans = trans_cpu.to(agent.device, non_blocking=True)
            done = done_cpu.to(agent.device)
            # The observation and the action "leads to it" are stored together.
            trans["action"] = act
            cache.append(trans.clone())
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            # Track success: if success at any step, mark episode as successful
            if "success" in trans:
                successes = torch.maximum(successes, trans["success"][:, 0] * (~once_done).float())
            once_done |= done
        cache = torch.stack(cache, dim=1)
        self.logger.scalar(f"episode/eval_score", returns.mean())
        self.logger.scalar(f"episode/eval_length", steps.to(torch.float32).mean())
        # Log success rate if available
        if "success" in cache:
            self.logger.scalar(f"episode/eval_success_rate", successes.mean())
        if "image" in cache:
            self.logger.video(f"eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log:
            initial = agent.get_initial_state(1)
            self.logger.video("eval_open_loop", tools.to_np(agent.video_pred(cache[:1, :self.batch_length], (initial["stoch"], initial["deter"]))))
        
        # Save eval video to disk and upload to S3 every k eval steps
        self._eval_count += 1
        if "image" in cache and self.eval_video_every > 0 and self._eval_count % self.eval_video_every == 0:
            video = tools.to_np(cache["image"][:1])  # (1, T, H, W, C)
            tools.save_eval_video(
                video=video,
                logdir=self.logdir,
                step=train_step,
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
            )
            
            # Compute and save saliency video if enabled
            if self.saliency_enabled:
                print("Computing saliency maps for evaluation episode...")
                try:
                    # Get first episode for saliency computation
                    episode_data = cache[:1]  # (1, T, ...)
                    T = episode_data.shape[1]
                    
                    # Compute saliency for each frame
                    actor_saliency_list = []
                    critic_saliency_list = []
                    
                    # Initialize state
                    stoch, deter = agent._wm.dynamics.initial(1)
                    prev_action = torch.zeros(1, self.act_dim, device=agent.device)
                    
                    H, W, C = episode_data['image'][0, 0].shape
                    
                    for t in range(T):
                        # Get observation for this frame
                        obs_t = {k: v[0, t] for k, v in episode_data.items()}
                        is_first = obs_t.get('is_first', torch.tensor(False, device=agent.device))
                        
                        # Reset state on episode boundary
                        if is_first.item() if is_first.dim() == 0 else is_first.squeeze().item():
                            stoch, deter = agent._wm.dynamics.initial(1)
                            prev_action = torch.zeros(1, self.act_dim, device=agent.device)
                        
                        if t % self.saliency_every_n == 0:
                            # Compute saliency for this frame
                            actor_sal, critic_sal = saliency.compute_saliency_step(
                                world_model=agent._wm,
                                actor_critic=agent._ac,
                                obs={k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                                     for k, v in obs_t.items()},
                                prev_stoch=stoch.squeeze(0),
                                prev_deter=deter.squeeze(0),
                                prev_action=prev_action.squeeze(0),
                                is_first=is_first,
                                preprocess_fn=agent.preprocess,
                                stride=self.saliency_stride,
                                mask_sigma=self.saliency_mask_sigma,
                                blur_sigma=self.saliency_blur_sigma,
                                act_discrete=agent.act_discrete,
                            )
                            
                            # Upsample to full resolution
                            actor_sal = saliency.upsample_saliency(actor_sal, (H, W))
                            critic_sal = saliency.upsample_saliency(critic_sal, (H, W))
                            
                            # Normalize
                            actor_sal = saliency.normalize_saliency(actor_sal)
                            critic_sal = saliency.normalize_saliency(critic_sal)
                        else:
                            # Reuse previous saliency
                            if len(actor_saliency_list) > 0:
                                actor_sal = actor_saliency_list[-1]
                                critic_sal = critic_saliency_list[-1]
                            else:
                                actor_sal = torch.zeros(H, W, device=agent.device)
                                critic_sal = torch.zeros(H, W, device=agent.device)
                        
                        actor_saliency_list.append(tools.to_np(actor_sal))
                        critic_saliency_list.append(tools.to_np(critic_sal))
                        
                        # Update state for next step
                        obs_for_step = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() < 4 else v 
                                        for k, v in obs_t.items()}
                        p_obs = agent.preprocess(obs_for_step)
                        embed = agent._wm.encoder(p_obs)
                        stoch, deter, _ = agent._wm.dynamics.obs_step(
                            stoch, deter, prev_action, embed,
                            is_first.unsqueeze(0) if is_first.dim() == 0 else is_first.unsqueeze(0)
                        )
                        if 'action' in obs_t:
                            prev_action = obs_t['action'].unsqueeze(0)
                    
                    # Stack saliency maps: (T, H, W) -> (1, T, H, W)
                    actor_saliency_np = np.stack(actor_saliency_list, axis=0)[np.newaxis, ...]
                    critic_saliency_np = np.stack(critic_saliency_list, axis=0)[np.newaxis, ...]
                    
                    # Save saliency video
                    tools.save_saliency_video(
                        video=video,
                        actor_saliency=actor_saliency_np,
                        critic_saliency=critic_saliency_np,
                        logdir=self.logdir,
                        step=train_step,
                        s3_bucket=self.s3_bucket,
                        s3_prefix=self.s3_prefix,
                        alpha=self.saliency_alpha,
                        colormap=self.saliency_colormap,
                    )
                    print("Saliency video saved successfully.")
                except Exception as e:
                    print(f"[WARNING] Saliency computation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Compute and save post-hoc decoder visualizations if enabled
            if self.posthoc_decoder_enabled:
                print("Training and rendering post-hoc decoder...")
                try:
                    # Lazy initialization of post-hoc decoder
                    if self._posthoc_decoder is None:
                        # Get image shape from episode data
                        sample_img = cache['image'][0, 0]
                        if sample_img.dim() == 3:
                            H, W, C = sample_img.shape
                        else:
                            _, H, W, C = sample_img.shape
                        
                        self._posthoc_decoder = PostHocDecoder(
                            feat_size=agent._wm.dynamics.feat_size,
                            image_shape=(H, W, C),
                            hidden_dim=self.posthoc_decoder_hidden_dim,
                            depth=self.posthoc_decoder_depth,
                            lr=self.posthoc_decoder_lr,
                            device=str(agent.device),
                        )
                        # Try to load existing checkpoint
                        ckpt_path = self.logdir / "posthoc_decoder.pt"
                        self._posthoc_decoder.load(ckpt_path)
                    
                    # Get first episode for training/visualization
                    episode_data = cache[:1]
                    T = episode_data.shape[1]
                    
                    # Collect training data: (images, stoch, deter) from the episode
                    images = episode_data['image']  # (1, T, H, W, C)
                    
                    # Re-run world model to get posterior states
                    stoch, deter = agent._wm.dynamics.initial(1)
                    prev_action = torch.zeros(1, self.act_dim, device=agent.device)
                    stoch_list, deter_list = [], []
                    
                    for t in range(T):
                        obs_t = {k: v[:, t] for k, v in episode_data.items()}
                        is_first = obs_t.get('is_first', torch.tensor([[False]], device=agent.device))
                        is_first_val = is_first.flatten()[0] if is_first.numel() > 0 else is_first
                        
                        if is_first_val.item():
                            stoch, deter = agent._wm.dynamics.initial(1)
                            prev_action = torch.zeros(1, self.act_dim, device=agent.device)
                        
                        p_obs = agent.preprocess(obs_t)
                        embed = agent._wm.encoder(p_obs)
                        stoch, deter, _ = agent._wm.dynamics.obs_step(
                            stoch, deter, prev_action, embed, is_first_val.unsqueeze(0)
                        )
                        stoch_list.append(stoch.clone())
                        deter_list.append(deter.clone())
                        
                        if 'action' in obs_t:
                            prev_action = obs_t['action']
                    
                    stoch_seq = torch.stack(stoch_list, dim=1)  # (1, T, stoch, discrete)
                    deter_seq = torch.stack(deter_list, dim=1)  # (1, T, deter)
                    
                    # Train post-hoc decoder for several steps
                    for _ in range(self.posthoc_decoder_train_steps):
                        # Sample random timesteps for training
                        batch_size = min(32, T)
                        t_indices = torch.randint(0, T, (batch_size,), device=agent.device)
                        
                        batch_images = images[0, t_indices]  # (batch, H, W, C)
                        batch_stoch = stoch_seq[0, t_indices]  # (batch, stoch, discrete)
                        batch_deter = deter_seq[0, t_indices]  # (batch, deter)
                        
                        metrics = self._posthoc_decoder.train_step(
                            images=batch_images,
                            stoch=batch_stoch.detach(),
                            deter=batch_deter.detach(),
                            world_model=agent._wm,
                        )
                    
                    # Log training metrics
                    for name, value in metrics.items():
                        self.logger.scalar(name, value)
                    
                    # Render posterior reconstructions
                    with torch.no_grad():
                        posterior_renders = self._posthoc_decoder.render_posterior(
                            stoch_seq[0], deter_seq[0], agent._wm
                        )  # (T, H, W, C)
                        
                        # Render imagination from last state
                        init_stoch = stoch_seq[0, -1]
                        init_deter = deter_seq[0, -1]
                        
                        # Generate random actions for imagination
                        imag_actions = torch.randn(
                            1, self.posthoc_decoder_imag_horizon, self.act_dim, 
                            device=agent.device
                        ).tanh()  # Bounded actions
                        
                        imag_renders = self._posthoc_decoder.render_imagination(
                            init_stoch.unsqueeze(0), init_deter.unsqueeze(0),
                            imag_actions, agent._wm
                        )  # (1, T_imag, H, W, C)
                    
                    # Save video
                    original_np = tools.to_np(images)  # (1, T, H, W, C)
                    posterior_np = tools.to_np(posterior_renders.unsqueeze(0))  # (1, T, H, W, C)
                    imag_np = tools.to_np(imag_renders)  # (1, T_imag, H, W, C)
                    
                    tools.save_posthoc_decoder_video(
                        original_video=original_np,
                        posterior_video=posterior_np,
                        imagination_video=imag_np,
                        logdir=self.logdir,
                        step=train_step,
                        s3_bucket=self.s3_bucket,
                        s3_prefix=self.s3_prefix,
                    )
                    
                    # Render counterfactuals if enabled
                    if self.posthoc_decoder_counterfactuals:
                        cf_renders = []
                        cf_labels = []
                        
                        for i in range(self.posthoc_decoder_num_cf_actions):
                            # Generate different random action sequences
                            cf_actions = torch.randn(
                                1, self.posthoc_decoder_imag_horizon, self.act_dim,
                                device=agent.device
                            ).tanh()
                            
                            with torch.no_grad():
                                cf_render = self._posthoc_decoder.render_imagination(
                                    init_stoch.unsqueeze(0), init_deter.unsqueeze(0),
                                    cf_actions, agent._wm
                                )
                            
                            cf_renders.append(tools.to_np(cf_render[0]))  # (T_imag, H, W, C)
                            cf_labels.append(f"Action seq {i+1}")
                        
                        # Save counterfactual grid
                        init_img = tools.to_np(images[0, -1])  # (H, W, C)
                        tools.save_counterfactual_grid(
                            init_image=init_img,
                            counterfactual_videos=cf_renders,
                            action_labels=cf_labels,
                            logdir=self.logdir,
                            step=train_step,
                            s3_bucket=self.s3_bucket,
                            s3_prefix=self.s3_prefix,
                        )
                    
                    # Save decoder checkpoint
                    self._posthoc_decoder.save(self.logdir / "posthoc_decoder.pt")
                    print("Post-hoc decoder visualization saved successfully.")
                    
                except Exception as e:
                    print(f"[WARNING] Post-hoc decoder failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # === Imagination Decoding Visualization ===
            # Run every K eval steps (K is large since this is expensive)
            if self.imag_decoding_enabled and self._eval_count % self.imag_decoding_every_k == 0:
                print(f"Running imagination decoding visualization (every {self.imag_decoding_every_k} evals)...")
                try:
                    # Ensure post-hoc decoder is initialized
                    if self._posthoc_decoder is None:
                        sample_img = cache['image'][0, 0]
                        if sample_img.dim() == 3:
                            H, W, C = sample_img.shape
                        else:
                            _, H, W, C = sample_img.shape
                        
                        self._posthoc_decoder = PostHocDecoder(
                            feat_size=agent._wm.dynamics.feat_size,
                            image_shape=(H, W, C),
                            hidden_dim=self.posthoc_decoder_hidden_dim,
                            depth=self.posthoc_decoder_depth,
                            lr=self.posthoc_decoder_lr,
                            device=str(agent.device),
                        )
                        ckpt_path = self.logdir / "posthoc_decoder.pt"
                        self._posthoc_decoder.load(ckpt_path)
                    
                    # Get episode data
                    episode_data = cache[:1]
                    T = episode_data.shape[1]
                    
                    # Need enough frames for context + prediction
                    required_len = self.imag_decoding_context_len + self.imag_decoding_pred_horizon
                    if T >= required_len:
                        # Sample a starting point for the visualization
                        # Pick multiple random starting points
                        for sample_idx in range(self.imag_decoding_num_samples):
                            start_idx = np.random.randint(0, max(1, T - required_len))
                            end_idx = start_idx + required_len
                            
                            # Extract context and future segments
                            K = self.imag_decoding_context_len
                            H_pred = min(self.imag_decoding_pred_horizon, T - start_idx - K)
                            
                            context_images = episode_data['image'][0, start_idx:start_idx+K]  # (K, H, W, C)
                            context_actions = episode_data['action'][0, start_idx:start_idx+K]  # (K, act_dim)
                            future_images = episode_data['image'][0, start_idx+K:start_idx+K+H_pred]  # (H_pred, H, W, C)
                            future_actions = episode_data['action'][0, start_idx+K:start_idx+K+H_pred]  # (H_pred, act_dim)
                            
                            if self.imag_decoding_show_uncertainty:
                                # Render with multiple samples to show uncertainty
                                with torch.no_grad():
                                    context_renders, future_samples = self._posthoc_decoder.render_open_loop_with_uncertainty(
                                        context_images=context_images,
                                        context_actions=context_actions,
                                        future_actions=future_actions,
                                        world_model=agent._wm,
                                        preprocess_fn=agent.preprocess,
                                        num_samples=self.imag_decoding_uncertainty_samples,
                                    )
                                
                                # Save uncertainty visualization
                                tools.save_uncertainty_visualization(
                                    context_true=tools.to_np(context_images),
                                    future_true=tools.to_np(future_images),
                                    future_samples=[tools.to_np(s) for s in future_samples],
                                    logdir=self.logdir,
                                    step=train_step,
                                    s3_bucket=self.s3_bucket,
                                    s3_prefix=self.s3_prefix,
                                )
                            else:
                                # Single sample open-loop prediction
                                with torch.no_grad():
                                    context_renders, future_renders, _ = self._posthoc_decoder.render_open_loop_prediction(
                                        context_images=context_images,
                                        context_actions=context_actions,
                                        future_actions=future_actions,
                                        world_model=agent._wm,
                                        preprocess_fn=agent.preprocess,
                                    )
                                
                                # Save as video
                                tools.save_open_loop_prediction_video(
                                    context_true=tools.to_np(context_images),
                                    context_rendered=tools.to_np(context_renders),
                                    future_true=tools.to_np(future_images),
                                    future_rendered=tools.to_np(future_renders),
                                    logdir=self.logdir,
                                    step=train_step,
                                    s3_bucket=self.s3_bucket,
                                    s3_prefix=self.s3_prefix,
                                )
                                
                                # Save as grid image
                                tools.save_open_loop_prediction_grid(
                                    context_true=tools.to_np(context_images),
                                    context_rendered=tools.to_np(context_renders),
                                    future_true=tools.to_np(future_images),
                                    future_rendered=tools.to_np(future_renders),
                                    logdir=self.logdir,
                                    step=train_step,
                                    s3_bucket=self.s3_bucket,
                                    s3_prefix=self.s3_prefix,
                                )
                            
                            # Only do first sample for grid/video, rest just for logging
                            break
                        
                        print("Imagination decoding visualization saved successfully.")
                    else:
                        print(f"[WARNING] Episode too short for imagination decoding ({T} < {required_len})")
                        
                except Exception as e:
                    print(f"[WARNING] Imagination decoding failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(envs.env_num, dtype=torch.int32, device=agent.device) # Increment this to prevent sampling across episode boundaries
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        act = agent_state["prev_action"].clone() # (B, A)
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video(f"train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar(f"episode/score", returns[i])
                        self.logger.scalar(f"episode/length", lengths[i])
                        self.logger.write(step + i) # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step +=  int((~done).sum()) * self._action_repeat # step is based on env side
            lengths += ~done
            # Step envs on CPU to avoid GPU<->CPU sync in the worker processes
            act_cpu = act.detach().to('cpu')
            done_cpu = done.detach().to('cpu')
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent
            trans = trans_cpu.to(agent.device, non_blocking=True)
            done = done_cpu.to(agent.device)
            # "agent_state" is initialized based on the "is_first" flag in trans.
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            # Store transition. The observation and the action "taken from it" are stored together.
            trans["action"] = act * ~done.unsqueeze(-1) # Mask action after done
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids # Don't lift dim
            if "image" in video_cache:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar(f"train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(f"{name}", tools.to_np(param))
                    self.logger.write(step, fps=True)
