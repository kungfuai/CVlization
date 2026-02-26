#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def _make_cheetah_env(image_shape=(224, 224)):
    """Minimal gymnasium wrapper around dm_control cheetah/run.

    Observation: concatenated position + velocity state vectors (not pixels).
    Render:      image_shape RGB frame via dm_control physics renderer.
    """
    import gymnasium as gym
    import numpy as np
    from dm_control import suite

    class CheetahRunEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self):
            self._env = suite.load("cheetah", "run")
            obs_size = sum(
                v.shape[0] if v.ndim else 1
                for v in self._env.reset().observation.values()
            )
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
            )
            spec = self._env.action_spec()
            self.action_space = gym.spaces.Box(
                spec.minimum.astype(np.float32),
                spec.maximum.astype(np.float32),
                dtype=np.float32,
            )
            self._image_shape = image_shape

        def _obs(self, ts):
            return np.concatenate(
                [v.flatten() for v in ts.observation.values()], dtype=np.float32
            )

        def reset(self, **kwargs):
            return self._obs(self._env.reset()), {}

        def step(self, action):
            ts = self._env.step(action)
            obs = self._obs(ts)
            done = ts.last()
            return obs, float(ts.reward or 0.0), done, done, {}

        def render(self):
            h, w = self._image_shape
            return self._env.physics.render(height=h, width=w, camera_id=0)

    return CheetahRunEnv()


def run_policy_rollout(asset_dir: Path, steps: int, num_envs: int, fps: int, device: str, video_out_dir: Path) -> None:
    import imageio
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ckpt     = asset_dir / "models" / "swm-dmc-expert-policies" / "cheetah" / "expert_policy" / "expert_policy.zip"
    vec_norm = asset_dir / "models" / "swm-dmc-expert-policies" / "cheetah" / "expert_policy" / "vec_normalize.pkl"
    require(ckpt)
    require(vec_norm)

    os.environ.setdefault("MUJOCO_GL", "egl")
    video_out_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([_make_cheetah_env])
    env = VecNormalize.load(str(vec_norm), env)
    env.training   = False
    env.norm_reward = False
    model = SAC.load(str(ckpt), env=env, device=device)

    gym_env = _make_cheetah_env()
    obs, _ = gym_env.reset()
    frames = []
    for _ in range(steps):
        frames.append(gym_env.render())
        obs_norm = env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, _, done, _, _ = gym_env.step(action[0])
        if done:
            obs, _ = gym_env.reset()
    gym_env.close()
    env.close()

    out = video_out_dir / "env_0.mp4"
    imageio.mimwrite(str(out), frames, fps=fps, codec="libx264",
                     output_params=["-crf", "20"])
    print(f"[ok] rollout complete. video: {out}")


def validate_world_model_load(asset_dir: Path) -> None:
    import inspect
    from stable_worldmodel.policy import AutoCostModel

    model_dir = asset_dir / "models" / "swm-dmc-cheetah"
    require(model_dir)
    ckpts = sorted(model_dir.glob("*_object.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No *_object.ckpt found in {model_dir}")
    latest = ckpts[-1]
    model_name = latest.name.removesuffix("_object.ckpt")

    try:
        sig = inspect.signature(AutoCostModel)
        if "run_name" in sig.parameters:
            model = AutoCostModel(str(model_dir))
        else:
            model = AutoCostModel(model_name, cache_dir=str(model_dir))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AutoCostModel could not unpickle *_object.ckpt because it depends on "
            f"missing legacy module path: {exc}. "
            "This indicates checkpoint/code-version mismatch (expected classes under "
            "top-level 'module')."
        ) from exc
    print(f"[ok] AutoCostModel loaded: {type(model).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stable-worldmodel DMControl parity checks with HF assets")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Asset directory created by download_assets.py",
    )
    parser.add_argument("--steps", type=int, default=200, help="Rollout steps")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized envs")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--device", type=str, default="cpu", help="Policy device (cpu|cuda)")
    parser.add_argument(
        "--video-out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "videos",
        help="Output directory for recorded videos",
    )
    parser.add_argument(
        "--skip-policy-rollout",
        action="store_true",
        help="Skip expert-policy rollout run",
    )
    parser.add_argument(
        "--validate-world-model-load",
        action="store_true",
        help="Attempt to load world-model checkpoint with AutoCostModel",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when optional checks fail",
    )
    args = parser.parse_args()

    failures = []

    if not args.skip_policy_rollout:
        try:
            run_policy_rollout(
                asset_dir=args.asset_dir,
                steps=args.steps,
                num_envs=args.num_envs,
                fps=args.fps,
                device=args.device,
                video_out_dir=args.video_out_dir,
            )
        except Exception as exc:
            msg = f"policy_rollout_failed: {exc}"
            failures.append(msg)
            print(f"[warn] {msg}")

    if args.validate_world_model_load:
        try:
            validate_world_model_load(args.asset_dir)
        except Exception as exc:
            msg = f"world_model_load_failed: {exc}"
            failures.append(msg)
            print(f"[warn] {msg}")

    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
