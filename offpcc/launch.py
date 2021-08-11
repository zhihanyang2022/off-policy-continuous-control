import gin
import argparse
import wandb
import os

import gym
from hac_pomdp_concat.domains import *
from gym.wrappers import RescaleAction

from basics.replay_buffer import ReplayBuffer
from basics.replay_buffer_recurrent import RecurrentReplayBufferGlobal
from basics.abstract_algorithms import OffPolicyRLAlgorithm, RecurrentOffPolicyRLAlgorithm
from algorithms import *
from algorithms_recurrent import *

# from basics.utils import get_device, set_random_seed
from basics.run_fns import train, make_log_dir, load_and_visualize_policy

algo_name2class = {
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
    'rdpg': RecurrentDDPG,
    'rtd3': RecurrentTD3,
    'rsac': RecurrentSAC,
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
        return RescaleAction(gym.make(args.env, rendering=args.render), -1, 1)


example_env = env_fn()


for run_id in args.run_id:  # args.run_id is a list of ints; could contain more than one run_ids

    # set_random_seed(seed=run_id, device=get_device())

    if args.algo.endswith('cnn'):
        algorithm = algo_name2class[args.algo](
            input_shape=example_env.observation_space.shape,
            action_dim=example_env.action_space.shape[0],
        )
    else:
        algorithm = algo_name2class[args.algo](
            input_dim=example_env.observation_space.shape[0],
            action_dim=example_env.action_space.shape[0],
        )

    if args.render:

        load_and_visualize_policy(
            env=example_env,
            algorithm=algorithm,
            log_dir=make_log_dir(args.env, args.algo, run_id),  # trained model will be loaded from here
            num_episodes=50,
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
            project="ant-heaven-hell-baselines",
            # entity='hainh22',
            group=f"timing-smooth-{args.algo} {args.env}",
            settings=wandb.Settings(_disable_stats=True),
            name=f's{run_id}',
        )

        # creating buffer based on the need of the algorithm
        if isinstance(algorithm, RecurrentOffPolicyRLAlgorithm):
            buffer = RecurrentReplayBufferGlobal(
                o_dim=example_env.observation_space.shape[0],
                a_dim=example_env.action_space.shape[0],
                max_episode_len=example_env.spec.max_episode_steps
            )
        elif isinstance(algorithm, OffPolicyRLAlgorithm):
            buffer = ReplayBuffer(
                input_shape=example_env.observation_space.shape,
                action_dim=example_env.action_space.shape[0]
            )

        train(
            env_fn=env_fn,
            algorithm=algorithm,
            buffer=buffer
        )

        run.finish()
