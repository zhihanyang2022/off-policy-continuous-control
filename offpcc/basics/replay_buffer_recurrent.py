import gin

from collections import namedtuple
import numpy as np
import torch

from basics.cuda_utils import get_device

RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')


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

        # placeholders

        self.pad_len = num_bptt - 1

        self.o = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, o_dim))
        self.a = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, a_dim))
        self.r = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))
        self.d = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))
        self.m = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))  # mask, in case episode_len < num_bptt
        self.ep_len = np.ones((capacity,)) * self.pad_len
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers

        self.episode_ptr = 0
        self.time_ptr = self.pad_len  # start filling at pad_len

        # trackers

        self.starting_new_episode = True

        # hyper-parameters

        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.num_bptt = num_bptt
        self.batch_size = batch_size

    def push(self, o, a, r, no, d, cutoff):

        # zero-out current slot at the beginning of an episode

        if self.starting_new_episode:

            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = self.pad_len
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # fill placeholders

        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # fill placeholders

            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.ep_len[self.episode_ptr] += self.num_bptt - 1  # for RHS padding
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = self.pad_len

            # update trackers

            self.starting_new_episode = True

        else:

            # update pointers

            self.time_ptr += 1

    @staticmethod
    def _as_probas(positive_values: np.array) -> np.array:
        return positive_values / np.sum(positive_values)

    @staticmethod
    def _prepare(np_array: np.array):
        return torch.tensor(np_array).float().to(get_device())

    def sample(self):

        # sample could take place in the middle of an episode
        # therefore, a partial episode could be sampled
        # this isn't incorrect because it is masked appropriately
        # nevertheless, I find it more elegant to avoid learning from them
        # because they could be very short

        # sample episode indices
        # assign higher probability to longer episodes

        options = np.where(self.ready_for_sampling == 1)[0]
        probas = self._as_probas(self.ep_len[options])
        ep_idxs = np.random.choice(options, p=probas, size=self.batch_size)

        # for selected episodes, get their length

        ep_lens = self.ep_len[ep_idxs]

        # to understand the following code, please first read
        # the example right above this section:
        # https://numpy.org/doc/stable/reference/arrays.indexing.html#combining-advanced-and-basic-indexing

        row_idxs_for_o = np.repeat(ep_idxs.reshape(-1, 1), repeats=self.num_bptt+1, axis=1)
        row_idxs_for_others = np.repeat(ep_idxs.reshape(-1, 1), repeats=self.num_bptt, axis=1)

        # suppose epi_idxs = [1, 2, 3, 4] and num_bptt = 10, then row_idxs =
        # 1 1 1 1 1 1 1 1 1 1
        # 2 2 2 2 2 2 2 2 2 2
        # 3 3 3 3 3 3 3 3 3 3
        # 4 4 4 4 4 4 4 4 4 4

        col_idxs_for_o = []
        col_idxs_for_others = []

        for ep_len in ep_lens:

            final_index = ep_len - 1  # the last valid index of the episode

            # first +1 is to correct for over subtraction
            # second +1 is to correct for the fact that np.random.randint does not include upper bound

            if ep_len > self.num_bptt:
                start_index = np.random.randint((final_index - self.num_bptt + 1) + 1)
            elif ep_len <= self.num_bptt:  # this is the case for which mask is actually useful
                start_index = 0

            end_index = start_index + (self.num_bptt - 1)  # correct for over addition

            col_idxs_for_o.append(np.arange(start_index, end_index + 1 + 1, 1))  # correct for not including upper bound
            col_idxs_for_others.append(np.arange(start_index, end_index + 1, 1))

        col_idxs_for_o = np.array(col_idxs_for_o)
        col_idxs_for_others = np.array(col_idxs_for_others)

        assert row_idxs_for_o.shape == col_idxs_for_o.shape == (self.batch_size, self.num_bptt + 1)
        assert row_idxs_for_others.shape == col_idxs_for_others.shape == (self.batch_size, self.num_bptt)

        # numpy advanced indexing

        o = self._prepare(self.o[row_idxs_for_o, col_idxs_for_o]).view(self.batch_size, self.num_bptt+1, self.o_dim)
        a = self._prepare(self.a[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.num_bptt, self.a_dim)
        r = self._prepare(self.r[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.num_bptt, 1)
        d = self._prepare(self.d[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.num_bptt, 1)
        m = self._prepare(self.m[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.num_bptt, 1)

        return RecurrentBatch(o, a, r, d, m)
