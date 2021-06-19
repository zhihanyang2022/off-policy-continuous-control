import gin

import torch
import torch.nn as nn
from sru import SRU


@gin.configurable(module=__name__)
class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_rnn_layers=2, use_sru=False):

        super().__init__()

        self.use_sru = use_sru

        if self.use_sru:
            rnn_klass = SRU  # should make training a little faster
        else:
            rnn_klass = nn.LSTM

        self.rnn = rnn_klass(input_dim, hidden_dim, num_layers=num_rnn_layers)

    def forward(self, observations, hidden=None, return_hidden=False):
        if not self.use_sru:
            self.rnn.flatten_parameters()
        observations = torch.swapaxes(observations, 0, 1)  # batch_first -> seq_len_first
        summary, hidden = self.rnn(observations, hidden)
        hidden = torch.swapaxes(observations, 0, 1)  # seq_len_first -> batch_first
        if return_hidden:
            return summary, hidden
        else:
            return summary
