import gym
import numpy as np


class DeepMindLabyrinth:
    """DeepMind Lab environment wrapper compatible with the R2Dreamer codebase.
    
    This wraps DeepMind Lab environments to provide a consistent interface
    with other environments like DMC.
    """
    
    metadata = {}

    ACTION_SET_DEFAULT = (
        (0, 0, 0, 1, 0, 0, 0),    # Forward
        (0, 0, 0, -1, 0, 0, 0),   # Backward
        (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),   # Look Right
        (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
        (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
        (0, 0, 0, 0, 1, 0, 0),    # Fire
    )

    ACTION_SET_MEDIUM = (
        (0, 0, 0, 1, 0, 0, 0),    # Forward
        (0, 0, 0, -1, 0, 0, 0),   # Backward
        (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
        (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),   # Look Right
        (0, 0, 0, 0, 0, 0, 0),    # Idle
    )

    ACTION_SET_SMALL = (
        (0, 0, 0, 1, 0, 0, 0),    # Forward
        (-20, 0, 0, 0, 0, 0, 0),  # Look Left
        (20, 0, 0, 0, 0, 0, 0),   # Look Right
    )

    # Mapping from action set name to action set
    ACTION_SETS = {
        'default': ACTION_SET_DEFAULT,
        'medium': ACTION_SET_MEDIUM,
        'small': ACTION_SET_SMALL,
    }

    def __init__(
        self,
        level,
        action_repeat=4,
        size=(64, 64),
        action_set='default',
        mode='train',
        level_cache=None,
        seed=0,
        runfiles_path=None,
    ):
        """Initialize DeepMind Lab environment.
        
        Args:
            level: Name of the DMLab30 level (without 'contributed/dmlab30/' prefix)
            action_repeat: Number of times to repeat each action
            size: Tuple of (height, width) for rendered images
            action_set: One of 'default', 'medium', 'small' or a custom tuple
            mode: Either 'train' or 'test'
            level_cache: Optional level cache for faster loading
            seed: Random seed for reproducibility
            runfiles_path: Optional path to DMLab runfiles
        """
        import deepmind_lab
        
        assert mode in ("train", "test")
        
        if runfiles_path:
            print("Setting DMLab runfiles path:", runfiles_path)
            deepmind_lab.set_runfiles_path(runfiles_path)
        
        self._config = {}
        self._config["width"] = size[1]
        self._config["height"] = size[0]
        self._config["logLevel"] = "WARN"
        
        if mode == "test":
            self._config["allowHoldOutLevels"] = "true"
            self._config["mixerSeed"] = 0x600D5EED
        
        self._action_repeat = action_repeat
        self._size = size
        self._random = np.random.RandomState(seed)
        
        # Build the full level path
        # DMLab30 levels are at: contributed/dmlab30/<level_name>
        # The level files are included when building DMLab from source with Bazel
        level_path = f"contributed/dmlab30/{level}"
        
        try:
            self._env = deepmind_lab.Lab(
                level=level_path,
                observations=["RGB_INTERLEAVED"],
                config={k: str(v) for k, v in self._config.items()},
                level_cache=level_cache,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DMLab level '{level_path}'. "
                f"Make sure DeepMind Lab is built with level data. "
                f"Try setting DMLAB_RUNFILES_PATH environment variable or "
                f"runfiles_path in config. Original error: {e}"
            ) from e
        
        # Resolve action set
        if isinstance(action_set, str):
            self._action_set = self.ACTION_SETS[action_set]
        else:
            self._action_set = action_set
            
        self._last_image = None
        self._done = True
        self._step_count = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        """Return observation space compatible with R2Dreamer."""
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            low=0, high=255, shape=self._size + (3,), dtype=np.uint8
        )
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return discrete action space."""
        return gym.spaces.Discrete(len(self._action_set))

    def reset(self):
        """Reset the environment and return initial observation."""
        self._done = False
        self._step_count = 0
        self._env.reset(seed=self._random.randint(0, 2**31 - 1))
        obs = self._get_obs()
        obs["is_terminal"] = False
        obs["is_first"] = True
        obs["is_last"] = False
        return obs

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._action_repeat)
        self._done = not self._env.is_running()
        self._step_count += 1
        
        obs = self._get_obs()
        obs["is_terminal"] = self._done
        obs["is_first"] = False
        obs["is_last"] = self._done
        
        info = {"discount": np.array(1.0 if not self._done else 0.0, np.float32)}
        
        return obs, reward, self._done, info

    def render(self, *args, **kwargs):
        """Render the environment."""
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._last_image

    def close(self):
        """Close the environment."""
        self._env.close()

    def _get_obs(self):
        """Get current observation dict."""
        if self._done:
            image = np.zeros_like(self._last_image)
        else:
            image = self._env.observations()["RGB_INTERLEAVED"]
        self._last_image = image
        return {"image": image}

