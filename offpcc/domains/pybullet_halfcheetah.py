import gym
import pybullet_envs
from domains.wrappers import FilterObsByIndex


def p():
    return FilterObsByIndex(gym.make("HalfCheetahBulletEnv-v0"), indices_to_keep=[0, 1, 2])
