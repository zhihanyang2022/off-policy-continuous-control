import gin
import wandb
import argparse

import numpy as np
import gym
from gym.wrappers import Monitor, RescaleAction

from cleanrl.basics.replay_buffer import ReplayBuffer
from cleanrl.basics.abstract_algorithms import OffPolicyRLAlgorithm
from algorithms import *

LOG_DIR = 'results'

def generate_log_dir(exp_name, run_id) -> str:
    return f'{LOG_DIR}/{exp_name}/{run_id}'

def test_for_one_episode(env, algorithm) -> tuple:
    state, done, episode_return, episode_len = env.reset(), False, 0, 0
    while not done:
        action = algorithm.act(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        episode_return += reward
        episode_len += 1
    return episode_return, episode_len

def visualize_trained_policy(
    env_fn,
    algorithm: OffPolicyRLAlgorithm,
    exp_name,
    run_id,
    num_episodes
) -> None:

    log_dir = generate_log_dir(exp_name, run_id)
    algorithm.load_actor(save_dir=log_dir, save_filename='actor.pth')

    for i in range(num_episodes):
        env = Monitor(
            env_fn(),
            directory=f'{log_dir}/videos/{i}'
        )
        test_for_one_episode(env, algorithm)

@gin.configurable(module=__name__)
def train(
    env_fn,
    algorithm: OffPolicyRLAlgorithm,
    buffer: ReplayBuffer,
    exp_name,  # TODO: explain that this can be different from env
    run_id,  # TODO: explain this
    num_epochs=None,  # TODO: all configurable arugments are default to None
    num_steps_per_epoch=None,
    update_every=None,  # number of environment interactions between gradient updates; however, the ratio of the two is locked to 1-to-1.
    num_test_episodes_per_epoch=None,
    update_after=None,  # for exploration
) -> None:

    env = env_fn()
    test_env = env_fn()

    state = env.reset()
    episode_len = 0

    """Follow from OpenAI Spinup's training loop style"""
    total_steps = num_steps_per_epoch * num_epochs

    for t in range(total_steps):

        if t >= update_after:   # num_exploration_steps have passed
            action = algorithm.act(state, deterministic=False)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        episode_len += 1

        # ignore the done flag if done is caused by hitting the maximum episode steps
        # TODO: environment needs to be wrapped by TimeLimit wrapper
        # however, the little catch is that the environment might actually be done
        # due to termination rather than timeout, but this is much less likely
        # so we just do it this way for convenience
        done = False if episode_len == env._max_episode_steps else True

        state = next_state

        # end of trajectory handling
        if done or (episode_len == env._max_episode_steps):
            # TODO: talk about termination handling
            state, episode_return, episode_len = env.reset(), 0, 0

        # update handling
        if t >= update_after and (t + 1) % update_every == 0:
            for j in range(update_every):
                batch = buffer.sample()
                algorithm.update_networks(batch)

        # end of epoch handling
        if (t + 1) % num_steps_per_epoch == 0:

            epoch = (t + 1) // num_steps_per_epoch
            episode_lens, episode_returns = [], []

            for j in range(num_test_episodes_per_epoch):
                episode_len, episode_return = test_for_one_episode(test_env, algorithm)
                episode_lens.append(episode_len)
                episode_returns.append(episode_return)

            wandb.log({
                'epoch': epoch,
                'test_mean_ep_len': np.mean(episode_lens),
                'test_mean_ep_ret': np.mean(episode_returns)
            })

    # TODO: save a csv and a model file after training

if __name__ == '__main__':

    algo_name2class = {
        'ddpg': DDPG,
        'sac': SAC
    }

    ON_POLICY_METHODS = ['A2C', 'TRPO', 'PPO']
    OFF_POLICY_METHODS_DISC = ['DQN', 'QRDQN', 'IQN']
    OFF_POLICY_METHODS_CONT = ['DDPG', 'TD3', 'SAC']

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--disc_or_cont', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, help='Task-specific hyperparameters')
    args = parser.parse_args()

    if args.discrete_or_continuous == 'disc':

        pass

    elif args.discrete_or_continuous == 'cont':

        env_fn = lambda: RescaleAction(gym.make(args.env), -1, 1)
        example_env = gym.make(args.env)

        algorithm = algo_name2class[args.algo](
            input_dim=example_env.observation_space.shape[0],
            action_dim=example_env.action_space.shape[0],
        )

        if args.algo in ON_POLICY_METHODS:
            raise NotImplementedError
        elif args.algo in OFF_POLICY_METHODS_CONT:
            buffer = ReplayBuffer()
        else:
            assert False, "Unknown algorithm"

        train(env_fn=env_fn, algorithm=algorithm, buffer=buffer, exp_name=args.exp_name, run_id=args.run_id)

    else:

        assert False, "Unknown option for disc_or_cont"