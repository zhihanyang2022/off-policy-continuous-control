# """
# Script for running spinup implementions.
# """
#
# import argparse
#
# import torch.nn as nn
# import gym
# import gin
#
# from spinup import ddpg_pytorch as ddpg
# from spinup import td3_pytorch as td3
# from spinup import sac_pytorch as sac
#
# from domains import *
#
# @gin.configurable(module=__name__)
# def train_ddpg(
#     env_fn,
#     logger_kwargs,
#     max_ep_len,
#     seed=0,
#     ac_kwargs={},
#     steps_per_epoch=gin.REQUIRED,
#     epochs=gin.REQUIRED,
#     replay_size=gin.REQUIRED,
#     gamma=gin.REQUIRED,
#     polyak=gin.REQUIRED,
#     lr=gin.REQUIRED,
#     batch_size=gin.REQUIRED,
#     update_after=gin.REQUIRED,
#     update_every=gin.REQUIRED,
#     act_noise=gin.REQUIRED,
#     num_test_episodes=10,
# ):
#     ddpg()
#
#
#
#
# @gin.configurable(module=__name__)
# def run_spinup(
#     env_id,
#
# ):
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--env')
# parser.add_argument('--algo')
#
#
#
# env_fn = lambda : gym.make('car-heaven-hell-concat-v0')
#
# ac_kwargs = dict(hidden_sizes=[256, 256], activation=nn.ReLU)
#
# logger_kwargs = dict(output_dir='data/ddpg_pendulum', exp_name='ddpg_pendulum')
#
# ddpg(
#     env_fn=env_fn,
#     ac_kwargs=ac_kwargs,
#     steps_per_epoch=1000,
#     num_test_episodes=10,
#     update_after=900,
#     start_steps=900,
#     epochs=200,
#     logger_kwargs=logger_kwargs,
#     max_ep_len=env_fn().spec.max_episode_steps
# )