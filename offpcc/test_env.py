import argparse
import gym
from domains import *
from pomdp_robot_domains import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, type=str)
parser.add_argument('--use_random_policy', action="store_true")
args = parser.parse_args()

env = gym.wrappers.RescaleAction(gym.make(args.env, rendering=True), -1, 1)

print('=> Env:', env)
print('=> Timeout:', env.spec.max_episode_steps)
print('=> Observation space:', env.observation_space)
print('=> Observation space low:', env.observation_space.low)
print('=> Observation space high:', env.observation_space.high)
print('=> Random trajectory:')

state = env.reset()
print(state)

ret = 0
cnt = 0
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    # state, reward, done, info = env.step([0])
    env.render()
    ret += reward
    cnt += 1
    if not (cnt == env.spec.max_episode_steps):
        print(state, reward, done)
    else:
        print(state, reward, done, info['TimeLimit.truncated'])
    if done:
        break

print('Reset:', env.reset())

print('Return:', ret)
print('Ep len:', cnt)