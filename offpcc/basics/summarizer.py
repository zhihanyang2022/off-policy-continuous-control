import gin

import torch.nn as nn


@gin.configurable(module=__name__)
class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_rnn_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_rnn_layers)

    def forward(self, observations, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary
