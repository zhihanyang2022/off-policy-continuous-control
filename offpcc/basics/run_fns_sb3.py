import gin
import numpy as np
import gym
from gym.wrappers import Monitor
import warnings
from typing import Any, Dict, Optional, Union
import wandb
import os

from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class MyEvalCallback(EventCallback):

    """
    Minimal callback class for logging to wandb
    Modified directly from
    https://github.com/DLR-RM/stable-baselines3/blob/88e1be9ff5e04b7688efa44951f845b7daf5717f/stable_baselines3/common/callbacks.py#L268
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        seed: int = 0,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.seed = seed

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            wandb.log({
                'epoch': self.n_calls % self.eval_freq,
                'timestep': self.n_calls,
                'test_ep_ret': np.mean(episode_rewards),
                'test_ep_len': np.mean(episode_lengths)
            })

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.
        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


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


BASE_LOG_DIR = '../results_sb3'


def make_log_dir(env_name, algo_name, seed) -> str:
    log_dir = f'{BASE_LOG_DIR}/{env_name}/{algo_name}/{seed}'
    return log_dir


def load_and_visualize_policy(
        env_fn,
        model,
        log_dir,
) -> None:
    model = model.load(os.path.join(log_dir, 'networks.zip'))
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env_fn(),
        n_eval_episodes=5,
        render=True,
        deterministic=True,
        return_episode_rewards=True,
    )
    print(episode_rewards, episode_lengths)
