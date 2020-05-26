"""
https://github.com/joe-siyuan-qiao/WeightStandardization
https://arxiv.org/abs/1903.10520

MIT License

Copyright (c) 2017 Wei Yang

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
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class WS_Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        super(WS_Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, 'zeros')

    def forward(self, x):
        w = self.weight
        
        w_mean = torch.mean(w, dim=(1,2), keepdim=True)
        w = w - w_mean

        w_std = w.view(w.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        w = w / w_std

        return F.conv1d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class WS_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WS_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def wsconv_gn_1d(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, gn_groups=32):
    mdl = nn.Sequential(WS_Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), 
                        nn.GroupNorm(gn_groups, out_channels))

    return mdl

def wsconv_gn_2d(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, num_groups=32):
    mdl = nn.Sequential(WS_Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), 
                        nn.GroupNorm(num_groups, out_channels))

    return mdl






