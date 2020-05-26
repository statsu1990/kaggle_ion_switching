"""
https://github.com/jadore801120/attention-is-all-you-need-pytorch

MIT License

Copyright (c) 2017 Victor Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = self.dropout(F.softmax(attn, dim=-1))

        return attn

class TimePositionAttention(nn.Module):
    """
    (batch, channel, n_position, time)
    """
    def __init__(self, nfeature, in_channel, n_position, dropout=0.1, gn_group=None):
        super(TimePositionAttention, self).__init__()

        def get_norm(_ch):
            if gn_group is None:
                return nn.BatchNorm1d(_ch)
            else:
                return nn.GroupNorm(gn_group, _ch)

        self.reduce_posi_ch = nn.Sequential(nn.Conv1d(in_channel*n_position, nfeature, kernel_size=1),
                                            get_norm(nfeature),
                                            nn.ReLU(inplace=True),)

        self.w_q = nn.Conv1d(nfeature, nfeature, kernel_size=1, bias=False)
        self.w_k = nn.Conv1d(nfeature, nfeature, kernel_size=1, bias=False)
        self.w_v = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.attention = ScaledDotProductAttention(nfeature**0.5, attn_dropout=dropout)

        self.fc = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        """
        Args:
            x : shape (batch, channel, n_position, time)
        """
        # (batch, channel, n_position, time) -> (batch, channel*n_position, time)
        xsize = x.size()
        h = x.view(xsize[0], xsize[1]*xsize[2], xsize[3])

        # (batch, nfeature, time)
        h = self.reduce_posi_ch(h)

        # q, k, v
        q = self.w_q(h).permute(0, 2, 1) # (batch, time, nfeature)
        k = self.w_k(h).permute(0, 2, 1) # (batch, time, nfeature)
        v = self.w_v(x) # (batch, channel, n_position, time)

        # attention
        attn = self.attention(q, k)  # (batch, time, time)
        attn = attn.unsqueeze(1)  # (batch, 1, time, time)

        output = torch.matmul(attn, v) # TODO : have bug
        h = self.dropout(self.fc(output))
        h = h + x
        return h


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn