"""
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter

import math

class SignalTransformer(nn.Module):
    def __init__(self, signal_encoder, positional_encoder, transformer_encoder, decoder):
        super(SignalTransformer, self).__init__()

        self.sig_encoder = signal_encoder
        self.pos_encoder = positional_encoder
        self.transformer_encoder = transformer_encoder
        self.decoder = decoder

        return

    def forward(self, x, trans_matrix=None):
        # signal encoder
        if self.sig_encoder is not None:
            h = self.sig_encoder(x, trans_matrix)
        else:
            h = x

        # positional encoder
        if self.pos_encoder is not None:
            h = self.pos_encoder(h)

        # transformer encoder
        if self.transformer_encoder is not None:
            h = self.transformer_encoder(h)
        h = h.permute(1, 0, 2) # (Length, Batch, feature) -> (Batch, Length, feature)

        # decoder
        if self.decoder is not None:
            h = self.decoder(h)

        # return
        return h

class PositionalEncoding(nn.Module):
    def __init__(self, nfeature, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nfeature)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nfeature, 2).float() * (-math.log(10000.0) / nfeature))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransEncorder(nn.Module):
    def __init__(self, nfeature, nhead, nhid, dropout, nlayers):
        super(TransEncorder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(nfeature, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, x):
        """
        Args:
            x: (length, batch size, feature)
        """
        return self.transformer_encoder(x)

class MyTransEncorder(nn.Module):
    def __init__(self, nfeature, nhead, nhid, dropout, nlayers, pre_norm=True, norm='ln', gn_group=None, qksame_attn=False):
        super(MyTransEncorder, self).__init__()
        encoder_layers = MyTransformerEncoderLayer(nfeature, nhead, nhid, dropout, pre_norm, norm, gn_group, qksame_attn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, x):
        """
        Args:
            x: (length, batch size, feature)
        """
        return self.transformer_encoder(x)

class MyTransformerEncoderLayer(nn.Module):
    """
    https://arxiv.org/pdf/2002.04745.pdf
    """
    def __init__(self, nfeature, nhead, nhid, dropout, pre_norm=True, norm='ln', gn_group=None, qksame_attn=False):
        super(MyTransformerEncoderLayer, self).__init__()
        self.pre_norm = pre_norm
        self.norm = norm

        if not qksame_attn:
            self.self_attn = nn.MultiheadAttention(nfeature, nhead, dropout=dropout)
        else:
            self.self_attn = MultiheadQKSameAttention(nfeature, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(nfeature, nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid, nfeature)
        self.activation = nn.ReLU(inplace=True)

        # normalization
        def get_norm():
            if norm == 'bn':
                return nn.BatchNorm1d(nfeature)
            elif norm == 'gn':
                return nn.GroupNorm(gn_group, nfeature)
            else:
                return nn.LayerNorm(nfeature)
        self.norm_layer1 = get_norm()
        self.norm_layer2 = get_norm()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        return

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (length, batch size, feature)
        """
        # attention
        if self.pre_norm:
            h1 = self.normalization(x, self.norm_layer1)
        else:
            h1 = x
        h1 = self.self_attn(h1, h1, h1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        h1 = self.dropout1(h1) + x
        if not self.pre_norm:
            h1 = self.normalization(h1, self.norm_layer1)

        # Feedforward
        if self.pre_norm:
            h2 = self.normalization(h1, self.norm_layer2)
        else:
            h2 = h1
        h2 = self.linear2(self.dropout(self.activation(self.linear1(h2))))
        h2 = self.dropout2(h2) + h1
        if not self.pre_norm:
            h2 = self.normalization(h2, self.norm_layer2)

        return h2

    def normalization(self, x, norm_layer):
        """
        Args:
            x: (length, batch size, feature)
        """
        if self.norm == 'bn' or self.norm == 'gn':
            h = x.permute(1, 2, 0) # (length, batch size, feature) -> (batch size, feature, length)
            h = norm_layer(h)
            h = h.permute(2, 0, 1) # (batch size, feature, length) -> (length, batch size, feature)
        else:
            h = norm_layer(x)

        return h

class MultiheadQKSameAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadQKSameAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qk_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.qk_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        return F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.qk_proj_weight, k_proj_weight=self.qk_proj_weight,
            v_proj_weight=self.v_proj_weight)

class LinearDecoder(nn.Module):
    def __init__(self, nfeature, n_label, scale=1.0):
        super(LinearDecoder, self).__init__()
        self.scale = scale
        self.linear = nn.Linear(nfeature, n_label)

    def forward(self, x):
        return self.linear(x) * self.scale