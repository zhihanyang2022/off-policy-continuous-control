import gin
import argparse
import wandb
import os

import gym
from domains import *
import pybullet_envs
from gym.wrappers import RescaleAction

from basics.replay_buffer import ReplayBuffer
from basics.replay_buffer_recurrent import RecurrentReplayBufferGlobal
from basics.abstract_algorithms import OffPolicyRLAlgorithm, RecurrentOffPolicyRLAlgorithm
from algorithms import *
from algorithms_recurrent import *

from basics.run_fns import train, make_log_dir, load_and_visualize_policy

algo_name2class = {
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
    'rdpg': RecurrentDDPG,
    'rtd3': RecurrentTD3,
    'rsac': RecurrentSAC,
    'rsac_s': RecurrentSACSharing
}

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--algo', type=str, required=True, help='Choose among ddpg, ddpg-lstm, td3, td3-lstm, sac and sac-lstm')
parser.add_argument('--run_id', nargs='+', type=int, required=True)
parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')
parser.add_argument('--render', action='store_true', help='Visualize a trained policy (no training happens)')
parser.add_argument('--record', action='store_true', help='Record a trained policy (no training happens)')

args = parser.parse_args()
assert not (args.render and args.record), "You should only set one of these two flags."

gin.parse_config_file(args.config)


def env_fn():
    """Any wrapper by default copies the observation and action space of its wrappee."""
    if args.env.startswith("bumps"):  # some of our custom envs require special treatment
        return RescaleAction(gym.make(args.env, rendering=args.render), -1, 1)
    else:
        return RescaleAction(gym.make(args.env), -1, 1)


example_env = env_fn()


for run_id in args.run_id:  # args.run_id is a list of ints; could contain more than one run_ids

    algorithm = algo_name2class[args.algo](
        input_dim=example_env.observation_space.shape[0],
        action_dim=example_env.action_space.shape[0],
    )

    if args.render:

        load_and_visualize_policy(
            env=example_env,
            algorithm=algorithm,
            log_dir=make_log_dir(args.env, args.algo, run_id),  # trained model will be loaded from here
            num_episodes=10,
            save_videos=False
        )

    elif args.record:

        load_and_visualize_policy(
            env=example_env,
            algorithm=algorithm,
            log_dir=make_log_dir(args.env, args.algo, run_id),  # trained model will be loaded from here
            num_episodes=10,
            save_videos=True
        )

    else:

        run = wandb.init(
            project=os.getenv('OFFPCC_WANDB_PROJECT'),
            entity=os.getenv('OFFPCC_WANDB_ENTITY'),
            group=f"{args.env} {args.algo} {args.config.split('configs/')[-1]} (ours)",
            settings=wandb.Settings(_disable_stats=True),
            name=f'run_id={run_id}',
            reinit=True
        )

        # creating buffer based on the need of the algorithm
        if isinstance(algorithm, RecurrentOffPolicyRLAlgorithm):  # TODO(future): change if new algorithms are added
            buffer = RecurrentReplayBufferGlobal(
                o_dim=example_env.observation_space.shape[0],
                a_dim=example_env.action_space.shape[0],
                max_episode_len=example_env.spec.max_episode_steps
            )
        elif isinstance(algorithm, OffPolicyRLAlgorithm):
            buffer = ReplayBuffer()

        train(
            env_fn=env_fn,
            algorithm=algorithm,
            buffer=buffer
        )

        run.finish()
