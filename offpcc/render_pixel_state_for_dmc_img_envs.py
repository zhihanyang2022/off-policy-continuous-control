import argparse
import gym
from domains import *

from basics.run_fns import test_for_one_episode

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, type=str)
args = parser.parse_args()


env = gym.wrappers.RescaleAction(gym.make(args.env), -1, 1)


class RandomAlgorithm:

    def __init__(self, env):
        self.env = env

    def act(self, *args, **kwargs):
        return self.env.action_space.sample()


print(test_for_one_episode(env, RandomAlgorithm(env), render=True, env_from_dmc=True, render_pixel_state=True))
