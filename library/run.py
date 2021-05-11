import gin
import wandb
import argparse

import gym
from gym.wrappers import RescaleAction

from basics.replay_buffer import ReplayBuffer
from algorithms import *

from basics.run_utils import train, visualize_trained_policy
from basics.run_utils import generate_log_dir

algo_name2class = {
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC
}

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--algo', type=str, required=True, help='Choose among "ddpg", "td3" and "sac"')
parser.add_argument('--run_id', type=int, required=True)
parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')
parser.add_argument('--visualize', action='store_true', help='Visualize a trained policy (no training happens)')  # default is false

args = parser.parse_args()

gin.parse_config_file(args.config)

log_dir = generate_log_dir(args.env, args.algo, args.run_id)

wandb.init(
    project='off-policy-continuous-control',
    entity='yangz2',
    group=f'{args.env}-{args.algo}',
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

# ==================================================

def env_fn():
    return RescaleAction(gym.make(args.env), -1, 1)

algorithm = algo_name2class[args.algo](
    input_dim=env_fn().observation_space.shape[0],
    action_dim=env_fn().action_space.shape[0],
)

if args.visualize:

    visualize_trained_policy(
        env_fn=env_fn,
        algorithm=algorithm,
        log_dir=log_dir,  # trained model will be loaded from here
        num_videos=10  # number of episodes to record
    )

else:

    buffer = ReplayBuffer()

    train(
        env_fn=env_fn,
        algorithm=algorithm,
        buffer=buffer,
        log_dir=log_dir  # the place to save training stats and trained model
    )
