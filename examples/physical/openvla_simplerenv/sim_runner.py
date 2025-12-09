"""
Simulation runner for OpenVLA + SimplerEnv evaluation.

Manages the ManiSkill2 environment and OpenVLA policy inference loop.
"""

import os
import sys
import logging

# Suppress verbose logging before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "torch", "jax"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from dataclasses import dataclass
from typing import Optional, Generator, Tuple, Dict, Any

import numpy as np


# Task registry with metadata for UI display
TASKS = {
    # Google Robot tasks (control at 3Hz)
    "google_robot_pick_coke_can": {
        "env_id": "google_robot_pick_coke_can",
        "description": "Pick up the Coke can (default orientation)",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_pick_horizontal_coke_can": {
        "env_id": "google_robot_pick_horizontal_coke_can",
        "description": "Pick up the horizontally placed Coke can",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_pick_vertical_coke_can": {
        "env_id": "google_robot_pick_vertical_coke_can",
        "description": "Pick up the vertically laid Coke can",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_pick_standing_coke_can": {
        "env_id": "google_robot_pick_standing_coke_can",
        "description": "Pick up the upright standing Coke can",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_move_near": {
        "env_id": "google_robot_move_near",
        "description": "Move an object near another object",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_open_drawer": {
        "env_id": "google_robot_open_drawer",
        "description": "Open the drawer",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_open_top_drawer": {
        "env_id": "google_robot_open_top_drawer",
        "description": "Open the top drawer",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_open_middle_drawer": {
        "env_id": "google_robot_open_middle_drawer",
        "description": "Open the middle drawer",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    "google_robot_close_drawer": {
        "env_id": "google_robot_close_drawer",
        "description": "Close the drawer",
        "embodiment": "google_robot",
        "control_freq": 3,
    },
    # WidowX / Bridge tasks (control at 5Hz)
    "widowx_spoon_on_towel": {
        "env_id": "widowx_spoon_on_towel",
        "description": "Place the spoon on the towel",
        "embodiment": "widowx",
        "control_freq": 5,
    },
    "widowx_carrot_on_plate": {
        "env_id": "widowx_carrot_on_plate",
        "description": "Place the carrot on the plate",
        "embodiment": "widowx",
        "control_freq": 5,
    },
    "widowx_stack_cube": {
        "env_id": "widowx_stack_cube",
        "description": "Stack the green cube on the yellow cube",
        "embodiment": "widowx",
        "control_freq": 5,
    },
    "widowx_put_eggplant_in_basket": {
        "env_id": "widowx_put_eggplant_in_basket",
        "description": "Put the eggplant in the basket",
        "embodiment": "widowx",
        "control_freq": 5,
    },
}


@dataclass
class StepResult:
    """Result from a single simulation step."""
    step: int
    image: np.ndarray  # RGB image (H, W, 3) uint8
    instruction: str
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class SimRunner:
    """
    Manages a SimplerEnv episode with OpenVLA policy inference.

    Provides a generator-based interface for streaming frames to a web client.
    """

    def __init__(
        self,
        task_id: str,
        model_path: str = "openvla/openvla-7b",
        seed: int = 0,
        max_steps: int = 200,
    ):
        """
        Initialize the simulation runner.

        Args:
            task_id: Task identifier from TASKS registry
            model_path: HuggingFace model path for OpenVLA
            seed: Random seed for environment
            max_steps: Maximum steps per episode
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")

        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self.model_path = model_path
        self.seed = seed
        self.max_steps = max_steps

        self.env = None
        self.policy = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of environment and policy (heavy imports)."""
        if self._initialized:
            return

        # Import SimplerEnv (triggers ManiSkill2 registration)
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference

        self._get_image_from_obs = get_image_from_maniskill2_obs_dict

        # Determine policy setup based on embodiment
        embodiment = self.task_config["embodiment"]
        if embodiment == "google_robot":
            policy_setup = "google_robot"
        elif embodiment == "widowx":
            policy_setup = "widowx_bridge"
        else:
            raise ValueError(f"Unknown embodiment: {embodiment}")

        print(f"Creating environment: {self.task_config['env_id']}")
        self.env = simpler_env.make(self.task_config["env_id"])

        print(f"Loading OpenVLA policy: {self.model_path}")
        self.policy = OpenVLAInference(
            saved_model_path=self.model_path,
            policy_setup=policy_setup,
        )

        self._initialized = True
        print("Simulation runner initialized")

    def run_episode(self) -> Generator[StepResult, None, Tuple[bool, Dict]]:
        """
        Run a single episode, yielding step results.

        Yields:
            StepResult for each simulation step

        Returns:
            Tuple of (success, episode_stats) at episode end
        """
        self._lazy_init()

        # Reset environment
        obs, reset_info = self.env.reset(seed=self.seed)
        instruction = self.env.get_language_instruction()
        is_final_subtask = self.env.is_final_subtask()

        # Reset policy with instruction
        self.policy.reset(instruction)

        # Get initial image
        image = self._get_image_from_obs(self.env, obs)

        # Yield initial state
        yield StepResult(
            step=0,
            image=image,
            instruction=instruction,
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
            info=reset_info,
        )

        # Run episode loop
        predicted_terminated = False
        success = False
        truncated = False

        for step in range(1, self.max_steps + 1):
            # Get action from policy
            raw_action, action = self.policy.step(image, instruction)

            # Check if policy predicts episode termination
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not is_final_subtask:
                # Advance to next subtask for long-horizon tasks
                predicted_terminated = False
                self.env.advance_to_next_subtask()

            # Step environment
            action_array = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"]
            ])
            obs, reward, success, truncated, info = self.env.step(action_array)

            # Update instruction if changed (for long-horizon tasks)
            new_instruction = self.env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                self.policy.reset(instruction)

            is_final_subtask = self.env.is_final_subtask()

            # Get new image
            image = self._get_image_from_obs(self.env, obs)

            # Yield step result
            yield StepResult(
                step=step,
                image=image,
                instruction=instruction,
                reward=reward,
                success=success,
                terminated=predicted_terminated,
                truncated=truncated,
                info=info,
            )

            # Check termination conditions
            if predicted_terminated or truncated or success:
                break

        # Return final stats
        episode_stats = info.get("episode_stats", {})
        return success, episode_stats

    def get_random_action_episode(self) -> Generator[StepResult, None, Tuple[bool, Dict]]:
        """
        Run an episode with random actions (for testing without model).

        Yields:
            StepResult for each simulation step
        """
        self._lazy_init_env_only()

        obs, reset_info = self.env.reset(seed=self.seed)
        instruction = self.env.get_language_instruction()
        image = self._get_image_from_obs(self.env, obs)

        yield StepResult(
            step=0,
            image=image,
            instruction=instruction,
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
            info=reset_info,
        )

        for step in range(1, self.max_steps + 1):
            action = self.env.action_space.sample()
            obs, reward, success, truncated, info = self.env.step(action)
            image = self._get_image_from_obs(self.env, obs)

            yield StepResult(
                step=step,
                image=image,
                instruction=instruction,
                reward=reward,
                success=success,
                terminated=False,
                truncated=truncated,
                info=info,
            )

            if truncated or success:
                break

        return success, info.get("episode_stats", {})

    def _lazy_init_env_only(self):
        """Initialize only the environment (for random action testing)."""
        if self.env is not None:
            return

        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

        self._get_image_from_obs = get_image_from_maniskill2_obs_dict
        self.env = simpler_env.make(self.task_config["env_id"])


def get_available_tasks() -> Dict[str, Dict]:
    """Return the task registry for UI display."""
    return TASKS


if __name__ == "__main__":
    # Simple test: run a few steps with random actions
    import argparse

    parser = argparse.ArgumentParser(description="Test SimRunner")
    parser.add_argument("--task", default="widowx_spoon_on_towel", choices=list(TASKS.keys()))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="Use random actions instead of OpenVLA")
    args = parser.parse_args()

    print(f"Testing task: {args.task}")
    runner = SimRunner(task_id=args.task, max_steps=args.steps)

    if args.random:
        episode_gen = runner.get_random_action_episode()
    else:
        episode_gen = runner.run_episode()

    for result in episode_gen:
        print(f"Step {result.step}: instruction='{result.instruction}', "
              f"reward={result.reward:.3f}, success={result.success}")
        if result.step >= args.steps:
            break

    print("Test complete")
