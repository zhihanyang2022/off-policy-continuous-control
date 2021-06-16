from basics.replay_buffer_recurrent import RecurrentReplayBufferGlobal
import gym
from domains import *


buffer = RecurrentReplayBufferGlobal(
    o_dim=2,
    a_dim=1,
    max_episode_len=200,
    capacity=100,
    batch_size=1
)

env = gym.make("cartpole-balance-pomdp-v0")

for i in range(10):
    state = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        buffer.push(state, action, reward, next_state, done, cutoff=False)
        if done:
            break
        state = next_state


print(buffer.sample())
