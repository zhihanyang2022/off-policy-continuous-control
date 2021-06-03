import gym
from domains import *
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn

env = gym.make('car-concat20-v0')

mode = "random"


def get_network():
    return nn.Sequential(
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


num_lefts_s = []
num_right_s = []

for i in tqdm(range(100)):  # 1000 trials

    num_lefts = 0
    num_rights = 0

    network = get_network()
    for j in range(10):  # 10 episodes
        state = env.reset()
        while True:
            state, reward, done, info = env.step(env.action_space.sample())
            # state, reward, done, info = env.step(act_using_network(state, network))
            if done:
                final_state = state[-3]
                if final_state >= 1:
                    num_rights += 1
                elif final_state <= -1:
                    num_lefts += 1
                break

    num_lefts_s.append(num_lefts)
    num_right_s.append(num_rights)

import numpy as np

arr = np.zeros((11, 11))
for num_lefts, num_rights in zip(num_lefts_s, num_right_s):
    arr[num_lefts, num_rights] += 1
print(arr)

plt.scatter(num_lefts_s, num_right_s, alpha=0.1)
plt.title('Out of 10 exploration episodes (100 trials)')
plt.xlabel('Num of hells reached')
plt.ylabel('Num of heavens reached')
plt.savefig('blabla')
