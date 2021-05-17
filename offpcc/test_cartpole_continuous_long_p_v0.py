import gym
from domains import *

env = gym.make('cartpole-continuous-long-v-concat-v0')
print(env)
print(env.observation_space)

state = env.reset()
print(state)

while True:
    state, reward, done, info = env.step(env.action_space.sample())
    print(state)
    if done:
        break
