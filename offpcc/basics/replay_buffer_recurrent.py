import gin

from collections import namedtuple
import numpy as np
import torch

from basics.cuda_utils import get_device

RecurrentBatch = namedtuple('RecurrentBatch', 'o a r no d m')


@gin.configurable(module=__name__)
class RecurrentReplayBuffer:

    def __init__(
        self,
        o_dim,
        a_dim,
        capacity=gin.REQUIRED,
        max_episode_len=gin.REQUIRED,
        num_bptt=gin.REQUIRED,
        batch_size=gin.REQUIRED
    ):

        # num_bptt

        self.o = np.zeros((capacity, max_episode_len, o_dim))
        self.a = np.zeros((capacity, max_episode_len, a_dim))
        self.r = np.zeros((capacity, max_episode_len, 1))
        self.no = np.zeros((capacity, max_episode_len, o_dim))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))  # mask
        self.ep_len = np.zeros((capacity,))

        self.episode_ptr = 0
        self.time_ptr = 0

        self.num_episodes = 0

        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.num_bptt = num_bptt
        self.batch_size = batch_size

    def push(self, o, a, r, no, d, cutoff):

        # fill

        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.no[self.episode_ptr, self.time_ptr] = no
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # update pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

            # empty next slot

            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.no[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0

        else:

            self.time_ptr += 1

    @staticmethod
    def _as_probas(positive_values: np.array) -> np.array:
        return positive_values / np.sum(positive_values)

    @staticmethod
    def _prepare(np_array: np.array):
        return torch.tensor(np_array).float().to(get_device())

    def sample(self):

        # sample episode indices
        # assign higher probability to longer episodes

        options = np.arange(self.num_episodes)
        probas = self._as_probas(self.ep_len[:self.num_episodes])
        ep_idxs = np.random.choice(options, p=probas, size=self.batch_size)

        # for selected episodes, get their length

        ep_lens = self.ep_len[ep_idxs]

        # to understand the following code, please first read
        # the example right above this section:
        # https://numpy.org/doc/stable/reference/arrays.indexing.html#combining-advanced-and-basic-indexing

        row_idxs = np.repeat(ep_idxs.reshape(-1, 1), repeats=self.num_bptt, axis=1)

        # suppose epi_idxs = [1, 2, 3, 4] and num_bptt = 10, then row_idxs =
        # 1 1 1 1 1 1 1 1 1 1
        # 2 2 2 2 2 2 2 2 2 2
        # 3 3 3 3 3 3 3 3 3 3
        # 4 4 4 4 4 4 4 4 4 4

        col_idxs = []
        for ep_len in ep_lens:

            if ep_len > self.num_bptt:  # low < high is a must for numpy
                start = np.random.randint(ep_len - self.num_bptt)
            elif ep_len < self.num_bptt:  # this is the case for which mask is actually useful
                start = 0

            end = start + self.num_bptt

            col_idxs.append(np.arange(start, end, 1))

        col_idxs = np.array(col_idxs)

        assert row_idxs.shape == col_idxs.shape == (self.batch_size, self.num_bptt)

        # numpy advanced indexing

        o = self._prepare(self.o[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, self.o_dim)
        a = self._prepare(self.a[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, self.a_dim)
        r = self._prepare(self.r[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, 1)
        no = self._prepare(self.no[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, self.o_dim)
        d = self._prepare(self.d[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, 1)
        m = self._prepare(self.m[row_idxs, col_idxs]).view(self.batch_size, self.num_bptt, 1)

        return RecurrentBatch(o, a, r, no, d, m)