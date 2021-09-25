import argparse
import gym
from domains import *
import pybullet_envs
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, type=str)
parser.add_argument('--use_random_policy', action="store_true")
args = parser.parse_args()

env = gym.wrappers.RescaleAction(gym.make(args.env), -1, 1)

print('=> Env:', env)
print('=> Timeout:', env.spec.max_episode_steps)
print('=> Observation space:', env.observation_space)
print('=> Observation space low:', env.observation_space.low)
print('=> Observation space high:', env.observation_space.high)
print('=> Random trajectory:')

# env.render()
state = env.reset()
print(state)

ret = 0
cnt = 0
while True:
    state, reward, done, info = env.step(env.action_space.sample())

    # state, reward, done, info = env.step([0])
    # env.render(mode='rgb_array')
    img = plt.imshow(env.render(mode='rgb_array'))
    plt.pause(0.01)  # Need min display time > 0.0.
    plt.draw()
    # if cnt > 30:
    #     time.sleep(10)
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