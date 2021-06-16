import gin

from collections import namedtuple
import numpy as np
import torch

from basics.utils import get_device


RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array):
    return torch.tensor(np_array).float().to(get_device())


@gin.configurable(module=__name__)
class RecurrentReplayBufferGlobal:

    """Use this version when num_bptt == max_episode_len"""

    def __init__(
        self,
        o_dim,
        a_dim,
        max_episode_len,  # this will also serve as num_bptt
        capacity=gin.REQUIRED,
        batch_size=gin.REQUIRED,
    ):

        # placeholders

        self.o = np.zeros((capacity, max_episode_len + 1, o_dim))
        self.a = np.zeros((capacity, max_episode_len, a_dim))
        self.r = np.zeros((capacity, max_episode_len, 1))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))
        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers

        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers

        self.starting_new_episode = True
        self.num_episodes = 0

        # hyper-parameters

        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.batch_size = batch_size

        self.max_episode_len = max_episode_len

    def push(self, o, a, r, no, d, cutoff):

        # zero-out current slot at the beginning of an episode

        if self.starting_new_episode:

            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0
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
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers

            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:

            # update pointers

            self.time_ptr += 1

    def sample(self):

        assert self.batch_size <= self.num_episodes, "Please increase update_after correspondingly."

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        ep_idxs = np.random.choice(options, p=probas_of_options, size=self.batch_size)

        # grab the corresponding numpy array
        # and save computational effort for lstm

        max_ep_len_in_batch = np.max(ep_lens_of_options)

        o = self.o[ep_idxs][:, :max_ep_len_in_batch+1, :]
        a = self.a[ep_idxs][:, :max_ep_len_in_batch, :]
        r = self.r[ep_idxs][:, :max_ep_len_in_batch, :]
        d = self.d[ep_idxs][:, :max_ep_len_in_batch, :]
        m = self.d[ep_idxs][:, :max_ep_len_in_batch, :]

        # convert to tensors on the right device

        o = as_tensor_on_device(o).view(self.batch_size, max_ep_len_in_batch+1, self.o_dim)
        a = as_tensor_on_device(a).view(self.batch_size, max_ep_len_in_batch, self.a_dim)
        r = as_tensor_on_device(r).view(self.batch_size, max_ep_len_in_batch, 1)
        d = as_tensor_on_device(d).view(self.batch_size, max_ep_len_in_batch, 1)
        m = as_tensor_on_device(m).view(self.batch_size, max_ep_len_in_batch, 1)

        return RecurrentBatch(o, a, r, d, m)
