import os
import argparse
import gin
import wandb

import gym
from domains import *
from gym.wrappers import RescaleAction
from basics.run_fns_sb3 import configure_ddpg, configure_td3, configure_sac, train_and_save_model, load_and_visualize_policy, make_log_dir

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--algo', type=str, required=True, help='Choose among ddpg, td3 and sac')
parser.add_argument('--seed', nargs='+', type=int, required=True)
parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')
parser.add_argument('--visualize', action='store_true', help='Visualize a trained policy (no training happens)')  # default is false

args = parser.parse_args()

gin.parse_config_file(args.config)

for seed in args.seed:

    def env_fn():
        return RescaleAction(gym.make(args.env), -1, 1)

    if args.algo == 'ddpg':
        model = configure_ddpg(env_fn=env_fn, seed=seed)
    elif args.algo == 'td3':
        model = configure_td3(env_fn=env_fn, seed=seed)
    elif args.algo == 'sac':
        model = configure_sac(env_fn=env_fn, seed=seed)
    else:
        raise NotImplementedError(f'Algorithm {args.algo} is not available.')

    if args.visualize:

        load_and_visualize_policy(
            env_fn=env_fn,
            model=model,
            log_dir=make_log_dir(args.env, args.algo, seed),
        )

    else:

        wandb.init(
            project=os.getenv('OFFPCC_WANDB_PROJECT'),
            entity=os.getenv('OFFPCC_WANDB_ENTITY'),
            group=f"{args.env} {args.algo} {args.config.split('/')[-1]} (sb3)",
            settings=wandb.Settings(_disable_stats=True),
            name=f'seed={seed}'
        )

        train_and_save_model(env_fn, model, seed)
