import gin

from collections import namedtuple
import numpy as np
import torch

from basics.utils import get_device


RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m')
RecurrentBatchWithStoredStates = namedtuple('RecurrentBatchWithStoredStates', 'o a r d m actor_stored_states critic_stored_states')


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
            capacity,
            batch_size,
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

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        probas = as_probas(self.ep_len[options])
        ep_idxs = np.random.choice(options, p=probas, size=self.batch_size)

        # grab the corresponding episodes

        o = as_tensor_on_device(self.o[ep_idxs]).view(self.batch_size, self.max_episode_len+1, self.o_dim)
        a = as_tensor_on_device(self.a[ep_idxs]).view(self.batch_size, self.max_episode_len, self.a_dim)
        r = as_tensor_on_device(self.r[ep_idxs]).view(self.batch_size, self.max_episode_len, 1)
        d = as_tensor_on_device(self.d[ep_idxs]).view(self.batch_size, self.max_episode_len, 1)
        m = as_tensor_on_device(self.m[ep_idxs]).view(self.batch_size, self.max_episode_len, 1)

        return RecurrentBatch(o, a, r, d, m)


class RecurrentReplayBufferLocal:

    """Use this version when num_bptt << max_episode_len"""

    def __init__(
            self,
            o_dim,
            a_dim,
            max_episode_len,
            capacity,
            num_bptt,
            num_burn_in=gin.REQUIRED,
            store_state=gin.REQUIRED,
            hidden_dim=gin.REQUIRED,
            store_state_style:gin.REQUIRED,
    ):

        # placeholders

        self.pad_len = (num_burn_in + num_bptt) - 1

        self.o = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len+1, o_dim))
        self.a = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, a_dim))
        self.r = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))
        self.d = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))
        self.m = np.zeros((capacity, self.pad_len+max_episode_len+self.pad_len, 1))  # mask, when episode_len < num_bptt

        if store_state:
            self.h_n = np.zeros((capacity, ))

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
        self.seq_len = num_burn_in + num_bptt
        self.batch_size = batch_size
        self.store_state = store_state

    def push(self, o, a, r, no, d, cutoff, hidden):

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
            self.ep_len[self.episode_ptr] += self.pad_len  # for RHS padding
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = self.pad_len

            # update trackers

            self.starting_new_episode = True

        else:

            # update pointers

            self.time_ptr += 1

    def sample(self):

        # sample could take place in the middle of an episode
        # therefore, a partial episode could be sampled
        # this isn't incorrect because it is masked appropriately
        # nevertheless, I find it more elegant to avoid learning from them
        # because they could be very short

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        probas = as_probas(self.ep_len[options] - 2 * self.pad_len)
        ep_idxs = np.random.choice(options, p=probas, size=self.batch_size)

        # for selected episodes, get their length

        ep_lens = self.ep_len[ep_idxs]

        # to understand the following code, please first read
        # the example right above this section:
        # https://numpy.org/doc/stable/reference/arrays.indexing.html#combining-advanced-and-basic-indexing

        row_idxs_for_o = np.repeat(ep_idxs.reshape(-1, 1), repeats=self.seq_len+1, axis=1)
        row_idxs_for_others = np.repeat(ep_idxs.reshape(-1, 1), repeats=self.seq_len, axis=1)

        # suppose epi_idxs = [1, 2, 3, 4] and num_bptt = 10, then row_idxs_for_others =
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

            start_index = np.random.randint((final_index - self.seq_len + 1) + 1)

            end_index = start_index + (self.seq_len - 1)  # correct for over addition

        # grab the corresponding numpy array
        # and save computational effort for lstm

        max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

        assert row_idxs_for_o.shape == col_idxs_for_o.shape == (self.batch_size, self.seq_len + 1)
        assert row_idxs_for_others.shape == col_idxs_for_others.shape == (self.batch_size, self.seq_len)

        # convert to tensors on the right device

        o = as_tensor_on_device(self.o[row_idxs_for_o, col_idxs_for_o]).view(self.batch_size, self.seq_len+1, self.o_dim)
        a = as_tensor_on_device(self.a[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.seq_len, self.a_dim)
        r = as_tensor_on_device(self.r[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.seq_len, 1)
        d = as_tensor_on_device(self.d[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.seq_len, 1)
        m = as_tensor_on_device(self.m[row_idxs_for_others, col_idxs_for_others]).view(self.batch_size, self.seq_len, 1)

        return RecurrentBatch(o, a, r, d, m)
