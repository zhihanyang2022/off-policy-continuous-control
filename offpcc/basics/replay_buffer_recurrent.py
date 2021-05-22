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

        # placeholders

        self.o = np.zeros((capacity, max_episode_len, o_dim))
        self.a = np.zeros((capacity, max_episode_len, a_dim))
        self.r = np.zeros((capacity, max_episode_len, 1))
        self.no = np.zeros((capacity, max_episode_len, o_dim))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))  # mask
        self.ep_len = np.zeros((capacity,))

        # pointers

        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers

        self.num_episodes = 0
        self.just_finished_an_episode = False

        # hyper-parameters

        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.num_bptt = num_bptt
        self.batch_size = batch_size

    def push(self, o, a, r, no, d, cutoff):

        # zero-out current slot at the beginning of an episode

        if self.just_finished_an_episode:

            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.no[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0

            self.just_finished_an_episode = False

        # fill

        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.no[self.episode_ptr, self.time_ptr] = no
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers

            if self.num_episodes < self.capacity:
                self.num_episodes += 1
            self.just_finished_an_episode = True

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
        # however, this shouldn't cause a problem because it is masked appropriately

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

            final_index = ep_len - 1  # the last valid index of the episode

            # first +1 is to correct for over subtraction
            # second +1 is to correct for the fact that np.random.randint does not include upper bound

            if ep_len >= self.num_bptt:
                start_index = np.random.randint((final_index - self.num_bptt + 1) + 1)
            elif ep_len < self.num_bptt:  # this is the case for which mask is actually useful
                start_index = 0

            end_index = start_index + (self.num_bptt - 1)  # correct for over addition

            col_idxs.append(np.arange(start_index, end_index+1, 1))  # correct for not including upper bound

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