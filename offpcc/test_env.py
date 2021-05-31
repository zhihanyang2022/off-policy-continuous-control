import argparse
import gym
from domains import *

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, type=str)
args = parser.parse_args()

env = gym.make(args.env)
print('=> Env:', env)
print('=> Timeout:', env.spec.max_episode_steps)
print('=> Observation space:', env.observation_space)
print('=> Observation space low:', env.observation_space.low)
print('=> Observation space high:', env.observation_space.high)
print('=> Random trajectory:')

state = env.reset()
print(state)

ret = 0
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    ret += reward
    env.render()
    print(state)
    if done:
        break

print(ret)