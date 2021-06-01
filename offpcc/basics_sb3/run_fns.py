# @@@@@ imports @@@@@

import gin
import numpy as np

import wandb
import os
import gym

from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from basics_sb3.callbacks import MyEvalCallback

# @@@@@ wrapper functions for gin to configure models from SB3 @@@@@


@gin.configurable(module=__name__)
def configure_ddpg(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        action_noise=gin.REQUIRED
):
    env = env_fn()
    model = DDPG(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=capacity,
        learning_starts=update_after,  # note that in SB3 exploration is not random; it uses the randomly init model
        batch_size=batch_size,
        tau=1-polyak,
        gamma=gamma,
        train_freq=(update_every, 'step'),
        gradient_steps=update_every,
        action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                       sigma=action_noise * np.ones(env.action_space.shape)),
        policy_kwargs={'net_arch': list(hidden_dimensions)},
        verbose=1,
        seed=seed,
        device='cpu',
    )
    return model


@gin.configurable(module=__name__)
def configure_td3(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        action_noise=gin.REQUIRED,
        target_noise=gin.REQUIRED,
        noise_clip=gin.REQUIRED,
        policy_delay=gin.REQUIRED
):
    env = env_fn()
    model = TD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        buffer_size=capacity,
        learning_starts=update_after,
        batch_size=batch_size,
        tau=1-polyak,
        gamma=gamma,
        train_freq=(update_every, 'step'),
        gradient_steps=update_every,
        action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                       sigma=action_noise * np.ones(env.action_space.shape)),
        policy_delay=policy_delay,
        target_policy_noise=target_noise,
        target_noise_clip=noise_clip,
        policy_kwargs={'net_arch': list(hidden_dimensions)},
        verbose=1,
        seed=seed,
        device='cpu',
    )
    return model


@gin.configurable(module=__name__)
def configure_sac(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        alpha=gin.REQUIRED,
        autotune_alpha=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED
):
    model = SAC(
        policy='MlpPolicy',
        env=env_fn(),
        learning_rate=lr,
        buffer_size=capacity,
        learning_starts=update_after,
        batch_size=batch_size,
        tau=1-polyak,
        gamma=gamma,
        train_freq=(update_every, 'step'),
        gradient_steps=update_every,
        action_noise=None,
        ent_coef=f'auto_{str(alpha)}' if autotune_alpha else alpha,  # e.g., 'auto_0.1' use 0.1 as init value
        target_entropy="auto",  # this will be the negative of dim of action space; not used if autotune_alpha is False
        policy_kwargs={'net_arch': list(hidden_dimensions)},
        verbose=1,
        seed=seed,
        device='cpu'
    )
    return model

# @@@@@ function for training and, once finished, saveing the model to wandb cloud @@@@@


@gin.configurable(module=__name__)
def train_and_save_model(
        env_fn,
        model,
        seed,
        num_steps_per_epoch=gin.REQUIRED,
        num_epochs=gin.REQUIRED,
        num_test_episodes_per_epoch=gin.REQUIRED
):
    my_eval_callback = MyEvalCallback(env_fn(),
                                      seed=seed,
                                      eval_freq=num_steps_per_epoch,
                                      n_eval_episodes=num_test_episodes_per_epoch)
    model.learn(
        total_timesteps=num_steps_per_epoch*num_epochs,
        callback=my_eval_callback
    )
    model.save(os.path.join(wandb.run.dir, 'networks.zip'))

# @@@@@ function for loading and visualizing trained model (plz download from wandb) @@@@@


BASE_LOG_DIR = '../results_sb3'


def make_log_dir(env_name, algo_name, seed) -> str:
    """You should download the networks.zip file from wandb and put it inside this folder."""
    log_dir = f'{BASE_LOG_DIR}/{env_name}/{algo_name}/{seed}'
    return log_dir


def load_and_visualize_policy(
        env_fn,
        model,
        log_dir,
) -> None:
    model = model.load(os.path.join(log_dir, 'networks.zip'))
    """
    gym.wrappers.Monitor(
                env_fn(),
                directory=os.path.join(log_dir, str(i+1)),
                force=True
            )
    """
    env = env_fn()
    for i in range(10):
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env=env,
            n_eval_episodes=1,
            render=True,
            deterministic=True,
            return_episode_rewards=True,
        )
        print(
            'Return:', round(episode_rewards[0], 2),
            'Length:', episode_lengths[0]
        )
