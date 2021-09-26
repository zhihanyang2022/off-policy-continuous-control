import gin

import torch.nn as nn


@gin.configurable(module=__name__)
class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, recurrent_type='lstm'):

        super().__init__()

        if recurrent_type == 'lstm':
            self.recurrent_module = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'rnn':
            self.recurrent_module = nn.RNN(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'gru':
            self.recurrent_module = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        else:
            raise ValueError(f"{recurrent_type} not recognized")

    def forward(self, observations, hidden=None, return_hidden=False):
        self.recurrent_module.flatten_parameters()
        summary, hidden = self.recurrent_module(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary
