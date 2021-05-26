import os
import argparse
import gin
import wandb

import gym
from gym.wrappers import RescaleAction

from basics.run_spinup_utils import train_ddpg, train_td3, train_sac

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--algo', type=str, required=True, help='Choose among "ddpg", "td3" and "sac"')
parser.add_argument('--seed', nargs='+', type=int, required=True)
parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')

args = parser.parse_args()

gin.parse_config_file(args.config)

for seed in args.seed:

    def env_fn():
        return RescaleAction(gym.make(args.env), -1, 1)

    wandb.init(
        project=os.getenv('OFFPCC_WANDB_PROJECT'),
        entity=os.getenv('OFFPCC_WANDB_ENTITY'),
        group=f"{args.env} {args.algo} (spinup)",
        settings=wandb.Settings(_disable_stats=True),
        name=f'seed={seed}'
    )

    if args.algo == 'ddpg':
        train_ddpg(
            env_fn=env_fn,
            seed=seed,
            max_steps_per_episode=env_fn().spec.max_episode_steps
        )
    elif args.algo == 'td':
        train_td3(
            env_fn=env_fn,
            seed=seed,
            max_steps_per_episode=env_fn().spec.max_episode_steps
        )
    elif args.algo == 'sac':
        train_sac(
            env_fn=env_fn,
            seed=seed,
            max_steps_per_episode=env_fn().spec.max_episode_steps
        )
    else:
        raise NotImplementedError(f'{args.algo} is not an available algorithm')
