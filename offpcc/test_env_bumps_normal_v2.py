import gym
from domains import *
from copy import deepcopy

from tqdm import tqdm

env = gym.wrappers.RescaleAction(gym.make("pbc-bumps-normal-pomdp-v0", rendering=False), -1, 1)
env2 = gym.wrappers.RescaleAction(gym.make("pbc-bumps-normal-pomdp-v0"), -1, 1)


print('=> Env:', env)
print('=> Timeout:', env.spec.max_episode_steps)
print('=> Observation space:', env.observation_space)
print('=> Observation space low:', env.observation_space.low)
print('=> Observation space high:', env.observation_space.high)
print('=> Random trajectory:')

left = 0
middle = 0
right = 0
for i in tqdm(range(100)):
    state = env.reset()
    if (env.ori_y_bump1 < 0) and (env.ori_y_bump2 > 0):
        middle += 1
    if (env.ori_y_bump1 < 0) and (env.ori_y_bump2 < 0):
        left += 1
    if (env.ori_y_bump1 > 0) and (env.ori_y_bump2 > 0):
        right += 1

print(left, middle, right)
