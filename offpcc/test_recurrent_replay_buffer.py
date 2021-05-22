import numpy as np

from basics.replay_buffer_recurrent import RecurrentReplayBuffer

buffer = RecurrentReplayBuffer(1, 1, capacity=5, max_episode_len=10, num_bptt=3, batch_size=5)

for i in range(20):  # more than capacity capacity
    for j in range(7):  # less than max_episode_len
        buffer.push(np.array([i]), np.array([i]), 0.5, np.array([i]), int(j == 6), int(j == 9))

print('Observations')
print(buffer.o.reshape(5, 10))

print('Actions')
print(buffer.a.reshape(5, 10))

print('Rewards')
print(buffer.r.reshape(5, 10))

print('Dones')
print(buffer.d.reshape(5, 10))

print('Masks')
print(buffer.m.reshape(5, 10))

print('Ep len')
print(buffer.ep_len)

print('Sample')
sample = buffer.sample()
print(sample.o.view(5, 3))
print(sample.a.view(5, 3))
print(sample.r.view(5, 3))
print(sample.d.view(5, 3))
print(sample.m.view(5, 3))
