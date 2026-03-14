import os
import sys
import pathlib
import warnings
import math
import hydra
import torch

from dreamer import Dreamer
from trainer import OnlineTrainer
from buffer import Buffer
from envs import make_envs
import tools

warnings.filterwarnings('ignore')
os.environ["MUJOCO_GL"] = "egl"
sys.path.append(str(pathlib.Path(__file__).parent))
#torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    weights = (logdir / "xxx.pth")
    if weights.exists():
        print("Loaded saved weights:", weights)
        agent.load_state_dict(torch.load(weights))

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs, agent.act_dim)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
