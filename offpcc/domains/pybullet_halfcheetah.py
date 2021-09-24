import gym
import pybullet_envs
from domains.wrappers import FilterObsByIndex


def p():
    return FilterObsByIndex(
        gym.make("HalfCheetahBulletEnv-v0"),
        indices_to_keep=[0, 1, 2, 3, 4, 5, 6, 7] + [8, 10, 12, 14, 16, 18] + [20, 21, 22, 23, 24, 25]
    )


def v():
    return FilterObsByIndex(
        gym.make("HalfCheetahBulletEnv-v0"),
        indices_to_keep=[0, 1, 2, 3, 4, 5, 6, 7] + [9, 11, 13, 15, 17, 19] + [20, 21, 22, 23, 24, 25]
    )
