from . import wrappers
from . import parallel


def make_envs(config):
    if config.vectorized:
        train_envs = make_vec_envs(config.task, config.device, config.env_num)
        eval_envs = make_vec_envs(config.task, config.device, config.eval_episode_num)
        obs_space = train_envs.observation_space
        act_space = train_envs.action_space
    else:
        env_constructor = lambda idx: lambda: make_env(config, idx)
        train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
        eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
        obs_space = train_envs.observation_space
        act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space

def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab
        import os

        action_set = getattr(config, 'action_set', 'default')
        mode = getattr(config, 'mode', 'train')
        # Check for runfiles path in config or environment variable
        runfiles_path = getattr(config, 'runfiles_path', None)
        if runfiles_path is None:
            runfiles_path = os.environ.get('DMLAB_RUNFILES_PATH', None)
        
        env = dmlab.DeepMindLabyrinth(
            level=task,
            action_repeat=config.action_repeat,
            size=tuple(config.size),
            action_set=action_set,
            mode=mode,
            seed=config.seed + id,
            runfiles_path=runfiles_path,
        )
        # DeepMind Lab has discrete actions, use OneHotAction wrapper
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    env = wrappers.Dtype(env)
    return env

def make_vec_envs(task, device, env_num):
    raise NotImplementedError
