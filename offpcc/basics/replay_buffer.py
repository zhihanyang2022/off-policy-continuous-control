import gin
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from basics.utils import get_device
import kornia
# import random
# from collections import deque

Transition = namedtuple('Transition', 's a r ns d')
Batch = namedtuple('Batch', 's a r ns d')


# @gin.configurable(module=__name__)
# class ReplayBuffer:
#     """Just a standard FIFO replay buffer."""
#
#     def __init__(self, input_shape, action_dim, capacity=int(1e6), batch_size=100):
#         self.capacity = capacity
#         self.memory = deque(maxlen=capacity)
#         self.batch_size = batch_size
#
#     def push(self, s, a, r, ns, d) -> None:
#         self.memory.appendleft(Transition(s, a, r, ns, d))
#
#     def sample(self) -> Batch:
#         assert len(self.memory) >= self.batch_size, "Please increase update_after to be >= batch_size"
#         transitions = random.choices(self.memory, k=self.batch_size)  # sampling WITH replacement
#         batch_raw = Batch(*zip(*transitions))
#         s = torch.tensor(batch_raw.s, dtype=torch.float).view(self.batch_size, -1)
#         a = torch.tensor(batch_raw.a, dtype=torch.float).view(self.batch_size, -1)
#         r = torch.tensor(batch_raw.r, dtype=torch.float).view(self.batch_size, 1)
#         ns = torch.tensor(batch_raw.ns, dtype=torch.float).view(self.batch_size, -1)
#         d = torch.tensor(batch_raw.d, dtype=torch.float).view(self.batch_size, 1)
#         return Batch(*list(map(lambda x: x.to(get_device()), [s, a, r, ns, d])))


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(get_device())


@gin.configurable(module=__name__)
class ReplayBuffer:

    """
    Replay buffer that works for both vector and image observations.
    Inspired by Spinning Up's buffer style.
    Augmentation modified from DrQ's style: https://github.com/denisyarats/drq/blob/master/replay_buffer.py.

    For future reference, note that in pytorch images are usually stored as (bs, depth, height, width).
    Therefore, we recommend that you store channel-first images rather than channel-last images.
    """

    def __init__(self, input_shape, action_dim, capacity=int(1e6), batch_size=100, use_aug_for_img=True):

        self.input_shape = input_shape
        self.action_dim = action_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.use_aug_for_img = use_aug_for_img

        assert len(self.input_shape) == 1 or len(self.input_shape) == 3  # vector or image (nothing else)

        self.s = np.empty((capacity, *input_shape), dtype=np.float32)
        self.a = np.empty((capacity, action_dim), dtype=np.float32)
        self.r = np.empty((capacity, 1), dtype=np.float32)
        self.ns = np.empty((capacity, *input_shape), dtype=np.float32)
        self.d = np.empty((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.num_transitions = 0

        if len(self.input_shape) == 3 and self.use_aug_for_img:

            # docs:
            # - https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
            # - https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=
            #   randomcrop#kornia.augmentation.RandomCrop

            # by default, the augmentation is different for each item in batch (of course, we would want this)

            self.augmentator = nn.Sequential(
                nn.ReplicationPad2d(4),
                kornia.augmentation.RandomCrop((self.input_shape[1], self.input_shape[2]))
            )

    def push(self, s, a, r, ns, d):

        assert s.shape == self.input_shape
        assert len(a) == self.action_dim

        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        if self.num_transitions < self.capacity:
            self.num_transitions += 1

    def sample(self):

        assert self.num_transitions >= self.batch_size

        indices = np.random.randint(self.num_transitions, size=self.batch_size)
        s = as_tensor_on_device(self.s[indices]).view(self.batch_size, *self.input_shape)
        a = as_tensor_on_device(self.a[indices]).view(self.batch_size, self.action_dim)
        r = as_tensor_on_device(self.r[indices]).view(self.batch_size, 1)
        ns = as_tensor_on_device(self.ns[indices]).view(self.batch_size, *self.input_shape)
        d = as_tensor_on_device(self.d[indices]).view(self.batch_size, 1)

        if len(self.input_shape) == 3 and self.use_aug_for_img:
            with torch.no_grad():
                s = self.augmentator(s)
                ns = self.augmentator(ns)

        return Batch(s, a, r, ns, d)
