import gin
import wandb

import numpy as np
from gym.wrappers import Monitor

from basics.replay_buffer import ReplayBuffer

BASE_LOG_DIR = 'results'


def generate_log_dir(env_name, algo_name, run_id) -> str:
    return f'{BASE_LOG_DIR}/{env_name}/{algo_name}/{run_id}'


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
        algorithm,
        log_dir,
        num_videos
) -> None:
    algorithm.load_actor(save_dir=log_dir)

    for i in range(num_videos):
        env = Monitor(
            env_fn(),
            directory=f'{log_dir}/videos/{i}'
        )
        test_for_one_episode(env, algorithm)


@gin.configurable(module=__name__)
def train(
        env_fn,
        algorithm,
        buffer: ReplayBuffer,
        log_dir,
        num_epochs,  # TODO: all configurable arugments are default to None
        num_steps_per_epoch,
        update_every,
        # number of environment interactions between gradient updates; however, the ratio of the two is locked to 1-to-1.
        num_test_episodes_per_epoch,
        update_after,  # for exploration
) -> None:
    env = env_fn()
    test_env = env_fn()

    state = env.reset()
    episode_len = 0

    """Follow from OpenAI Spinup's training loop style"""
    total_steps = num_steps_per_epoch * num_epochs

    for t in range(total_steps):

        if t >= update_after:  # num_exploration_steps have passed
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

    algorithm.save_actor(log_dir)
