"""
run_sb3_utils.py
Originally written by Hai Nguyen
Modified by Zhihan Yang
"""

import gin

import gym
import numpy as np
import os
from argparse import ArgumentParser

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.td3.policies import TD3Policy

# import wandb
#
#
#
#             env = gym.make(args.env, rendering=True)
#         else:
#             env = gym.make(args.env)
#
#
#         wandb.init(project='pomdpr', settings=wandb.Settings(_disable_stats=True),
#                    group=args.group if args.group is not None else '_'.join(
#                        ["baselines", str(args.agent), str(args.env), str(args.seed)]),
#                    name=args.name if args.name is not None else 's' + str(args.seed))
#
#
#
#
#     if not args.rendering:
#         if agent in ['ddpg', 'td3']:
#             # The noise objects for DDPG/TD3
#             n_actions = env.action_space.shape[-1]
#             action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#
#         if agent in ['ddpg']:
#             model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, seed=args.seed)
#
#         if agent in ['td3']:
#             model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, seed=args.seed)
#
#         if agent in ['sac']:
#             model = SAC("MlpPolicy", env, verbose=1, seed=args.seed)
#
#     if not args.rendering:
#         model.learn(total_timesteps=args.timesteps, callback=eval_callback)
#         model.save(model_dir)


@gin.configurable(module=__name__)
def train_ddpg(
        env_fn,
        seed,
        hidden_dimensions=gin.REQUIRED,
        num_steps_per_epoch=gin.REQUIRED,
        num_epochs=gin.REQUIRED,
        capacity=gin.REQUIRED,
        gamma=gin.REQUIRED,
        polyak=gin.REQUIRED,
        lr=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        update_after=gin.REQUIRED,
        update_every=gin.REQUIRED,
        action_noise=gin.REQUIRED,
        num_test_episodes_per_epoch=gin.REQUIRED
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
        optimize_memory_usage=False,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs={'net_arch': hidden_dimensions},
        verbose=1,
        seed=seed,
        device='cpu',
        _init_setup_model=True
    )
    my_eval_callback = MyEvalCallback(env_fn(),
                                      seed=seed,
                                      eval_freq=num_steps_per_epoch,
                                      n_eval_episodes=num_test_episodes_per_epoch)
    model.learn(
        total_timesteps=num_steps_per_epoch*num_epochs,
        callback=my_eval_callback
    )