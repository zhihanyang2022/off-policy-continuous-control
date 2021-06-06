import gin

import torch.nn as nn

class Summarizer(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_rnn_layers=gin.REQUIRED,
    ):
        # TODO(zhihan): in the future, make more rnn types available (vanilla, lstm, gru)
        # TODO(zhihan): in the future, add CNN for image input
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_rnn_layers)

    def forward(self, observations, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary
