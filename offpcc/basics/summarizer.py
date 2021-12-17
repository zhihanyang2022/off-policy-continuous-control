import gin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = x + self._ff_block(x)

        return x



class PositionalEncoding(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 # dropout: float,
                 maxlen: int):
        super(PositionalEncoding, self).__init__()

        maxlen = maxlen + 1  # if maxlen is episode length, then the number of observations we store is maxlen + 1

        den = torch.exp(- torch.arange(0, hidden_dim, 2) * math.log(10000) / hidden_dim)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, hidden_dim))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, maxlen, obs_size)

        self.register_buffer('pos_embedding', pos_embedding)  # it will be saved as model parameters, but not updated

    def forward(self, embedded):
        return embedded + self.pos_embedding[:, :embedded.size(1), :]


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=get_device())) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerSummarizer(nn.Module):

    def __init__(self, obs_size, hidden_dim, max_len):
        super(TransformerSummarizer, self).__init__()

        self.input_projector = nn.Linear(obs_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=256, nhead=8,
            dim_feedforward=512, dropout=0,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 4)

    def forward(self, observations, prev_observations=None):

        # observations may have shape:
        # 1. (bs, max_len, obs_size) - in update_networks
        # 2. (1, 1, obs_size) - in this case, we need prev_observations (1, seq_len, obs_size), if it is not None

        if prev_observations is not None:
            x = torch.cat([prev_observations, observations], dim=1)
        else:
            x = observations

        x = self.input_projector(x)
        x = self.positional_encoding(x)
        mask = generate_square_subsequent_mask(x.size()[1])
        summary = self.transformer_encoder(src=x, mask=mask)  # (bs, maxlen or seq_len, hidden_dim)

        # we can safely squeeze here because only in the second case would the squeeze be in effect
        # i.e., maxlen > 1

        # Case 1: squeeze does nothing
        # Case 2: (1, hidden_size), which is perfect for inputting into the MLP actors and critics

        return summary[:, -observations.size()[1]:, ]
