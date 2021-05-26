import argparse

import torch.nn as nn
import gym
import gin

from spinup import ddpg_pytorch as ddpg
from spinup import td3_pytorch as td3
from spinup import sac_pytorch as sac

from domains import *

@gin.configurable(module=__name__)
def train_ddpg(
    env_fn,
    logger_kwargs,
    max_ep_len,
    seed=0,
    ac_kwargs={},
    steps_per_epoch=gin.REQUIRED,
    epochs=gin.REQUIRED,
    replay_size=gin.REQUIRED,
    gamma=gin.REQUIRED,
    polyak=gin.REQUIRED,
    lr=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    update_after=gin.REQUIRED,
    update_every=gin.REQUIRED,
    act_noise=gin.REQUIRED,
    num_test_episodes=10,
):
    ddpg()

# @gin.configurable()