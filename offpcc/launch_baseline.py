# @@@@@ imports @@@@@

import os
import argparse
import gin
import wandb

import gym
from domains import *
import pybullet_envs
from gym.wrappers import RescaleAction
from basics_sb3.run_fns import configure_ddpg, configure_td3, configure_sac, train_and_save_model, \
    load_and_visualize_policy, make_log_dir

# @@@@@ parse command line arguments @@@@@

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--algo', type=str, required=True, help='Choose among ddpg, td3 and sac')
parser.add_argument('--seed', nargs='+', type=int, required=True)
parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')
parser.add_argument('--render', action='store_true', help='Visualize a trained policy (no training happens)')
parser.add_argument('--record', action='store_true', help='Record a trained policy (no training happens)')

args = parser.parse_args()

# @@@@@ run each seed @@@@@

gin.parse_config_file(args.config)


def env_fn():
    return RescaleAction(gym.make(args.env), -1, 1)


for seed in args.seed:

    if args.algo == 'ddpg':
        model = configure_ddpg(env_fn=env_fn, seed=seed)
    elif args.algo == 'td3':
        model = configure_td3(env_fn=env_fn, seed=seed)
    elif args.algo == 'sac':
        model = configure_sac(env_fn=env_fn, seed=seed)
    else:
        raise NotImplementedError(f'Algorithm {args.algo} is not available from stable-baselines3.')

    if args.render:

        load_and_visualize_policy(
            env_fn=env_fn,
            model=model,
            log_dir=make_log_dir(args.env, args.algo, seed),
            num_episodes=10,
            save_videos=False
        )

    elif args.record:

        load_and_visualize_policy(
            env_fn=env_fn,
            model=model,
            log_dir=make_log_dir(args.env, args.algo, seed),
            num_episodes=10,
            save_videos=True
        )

    else:

        run = wandb.init(
            project="hierarchy_baselines",
            entity='hainh22',
            group=f"{args.env} {args.algo} {args.config.split('/')[-1]} (sb3)",
            settings=wandb.Settings(_disable_stats=True),
            name=f'seed={seed}',
            reinit=True  # allows to re-init multiple runs
        )

        train_and_save_model(env_fn, model, seed)

        run.finish()  # if I don't include this, then the next run is tracked by the current run
