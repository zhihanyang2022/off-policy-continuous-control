import gin
import math
import torch
import torch.nn as nn
from basics.utils import get_device


@gin.configurable(module=__name__)
class Summarizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2, recurrent_type='lstm'):

        super().__init__()

        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        else:
            raise ValueError(f"{recurrent_type} not recognized")

    def forward(self, observations, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary


# from https://pytorch.org/tutorials/beginner/translation_transformer.html


class PositionalEncoding(nn.Module):
    def __init__(self,
                 obs_size: int,
                 # dropout: float,
                 maxlen: int):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, obs_size, 2) * math.log(10000) / obs_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, obs_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, maxlen, obs_size)

        # self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)  # so that it will be saved

    def forward(self, observations):
        # return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        # observations has shape (bs, seq_len, obs_size)
        return observations + self.pos_embedding[:, :observations.size(1), :]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=get_device())) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerSummarizer(nn.Module):

    def __init__(self, obs_size, hidden_dim, max_len):
        super(TransformerSummarizer, self).__init__()
        self.positional_encoding = PositionalEncoding(obs_size, max_len)
        self.pre_mlp = nn.Linear(obs_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dropout=0,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)

    def forward(self, observations, prev_observations=None):

        # observations may have shape:
        # 1. (bs, max_len, obs_size) - in update_networks
        # 2. (1, 1, obs_size) - in this case, we need prev_observations (1, seq_len, obs_size), if it is not None

        if prev_observations is not None:
            observations = torch.cat([prev_observations, observations], dim=1)

        observations = self.positional_encoding(observations)
        mask = generate_square_subsequent_mask(observations.size()[1])
        summary = self.transformer_encoder(src=self.pre_mlp(observations), mask=mask)  # (bs, maxlen or seq_len, hidden_dim)

        # we can safely squeeze here because only in the second case would the squeeze be in effect
        # i.e., maxlen > 1

        # Case 1: squeeze does nothing
        # Case 2: (1, hidden_size), which is perfect for inputting into the MLP actors and critics

        return summary[:, -observations.size()[1]:, ].squeeze()
