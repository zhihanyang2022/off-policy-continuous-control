import argparse
import gym
from domains import *

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, type=str)
args = parser.parse_args()

env = gym.make(args.env)
print('=> Env:', env)
print('=> Observation space:', env.observation_space)
print('=> Observation space low:', env.observation_space.low)
print('=> Observation space high:', env.observation_space.high)
print('=> Random trajectory:')

state = env.reset()
print(state)

while True:
    state, reward, done, info = env.step(env.action_space.sample())
    print(state)
    if done:
        break
