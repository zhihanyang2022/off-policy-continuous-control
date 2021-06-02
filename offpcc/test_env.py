import argparse
import gym
from domains import *
import torch
import torch.nn as nn

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

state = env.reset()
print(state)

network = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, env.action_space.shape[0]),
    nn.Tanh()
)


def act_using_network(state, network):
    with torch.no_grad():
        return network(torch.FloatTensor(state).unsqueeze(0)).view(-1).cpu().numpy()


ret = 0
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(act_using_network(state, network))
    ret += reward
    env.render()
    print(state)
    if done:
        break

print(ret)