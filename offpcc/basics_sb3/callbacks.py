import wandb
import warnings
from typing import Any, Dict, Optional, Union

import gym
import numpy as np

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