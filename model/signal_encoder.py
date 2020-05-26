import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import weight_standalization as ws
from . import mixup as mu
from . import attention

class SEModule(nn.Module):
    def __init__(self, channels, reduction, dim_is_1d=True):
        super(SEModule, self).__init__()
        if dim_is_1d:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1,
                                 padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1,
                                 padding=0)
            self.sigmoid = nn.Sigmoid()
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                                 padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                                 padding=0)
            self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class ResConv2d(nn.Module):
    def __init__(self, kernel_sizes=[(3, 5), (3, 5), (3, 5)], 
                 channel=64, reduce_dim=1, 
                 gn_group=None, se_reduction=4, 
                 onedim_to_twodim=True, in_channel=1):
        super(ResConv2d, self).__init__()
        self.reduce_dim = reduce_dim
        self.onedim_to_twodim = onedim_to_twodim

        def _get_conv_norm(_in_ch, _out_ch, _ks, _n_gr):
            pad = (_ks[0]//2, _ks[1]//2)
            if _n_gr is None:
                mdl = conv_bn_2d(_in_ch, _out_ch, _ks, stride=1, padding=pad)
            else:
                mdl = conv_gn_2d(_in_ch, _out_ch, _ks, stride=1, padding=pad, num_groups=_n_gr)
            return mdl

        # conv layer
        self.conv_layers = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        in_ch = in_channel
        for i, ks in enumerate(kernel_sizes):
            self.conv_layers.append(nn.Sequential(_get_conv_norm(in_ch, channel, ks, gn_group), nn.ReLU(inplace=True)))
            if i == 0:
                self.se_layers.append(None)
            else:
                if se_reduction is None:
                    self.se_layers.append(None)
                else:
                    self.se_layers.append(SEModule(channel, se_reduction, dim_is_1d=False))

            in_ch = channel

        # reduce layer
        if self.reduce_dim == 1:
            self.reduce_layer = nn.Conv2d(channel, 1, 1)

    def forward(self, x, label=None):
        """
        Args:
            x: shape (batch, C, Length)
        """
        # adjust shape
        if self.onedim_to_twodim:
            h = x.unsqueeze(1) # (batch, C, Length) -> (batch, 1, C, Length)
        else:
            h = x

        # conv layer
        for i, (conv, se) in enumerate(zip(self.conv_layers, self.se_layers)):
            if i == 0:
                h = conv(h)
            else:
                if se is None:
                    h = h + conv(h)
                else:
                    h = h + se(conv(h))

        # reduce
        if self.reduce_dim == 1:
            h = self.reduce_layer(h)
            hsize = h.size()
            h = h.view((hsize[0], hsize[2], hsize[3]))
        else:
            h = torch.mean(h, dim=self.reduce_dim)

        return h

class SignalEncoder(nn.Module):
    def __init__(self, encoders):
        super(SignalEncoder, self).__init__()
        self.encoders = encoders

    def forward(self, x, trans_matrix):
        """
        Args:
            x : shape (Batch, C, Length)
        """
        # encoders
        h = x

        if hasattr(self.encoders, "__iter__"):
            for enc in self.encoders:
                h = enc(h, trans_matrix)
        else:
            h = self.encoders(h, trans_matrix)

        h = h.permute(2, 0, 1) # (Length, Batch, feature)

        return h

class PreActFilmBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn_group, se_reduction, use_film=False):
        super(PreActFilmBlock, self).__init__()
        self.use_film = use_film

        self.gn1 = nn.GroupNorm(gn_group, out_channels, affine=not use_film)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.gn2 = nn.GroupNorm(gn_group, out_channels, affine=not use_film)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        if se_reduction is not None:
            self.se_module = SEModule(out_channels, se_reduction)
        else:
            self.se_module = None

    def forward(self, x, sigma=None, mu=None):
        h = F.relu(self.gn1(x))
        h = self.conv1(h)
        h = self.gn2(h)
        if self.use_film:
            h = h * sigma.view(sigma.size()[0],-1,1) + mu.view(mu.size()[0],-1,1)
        h = self.conv2(F.relu(h))

        if self.se_module is None:
            h = h + x
        else:
            h = self.se_module(h) + x
        return h

class FilmBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn_group, se_reduction, use_film=False):
        super(FilmBlock, self).__init__()
        self.use_film = use_film

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.gn1 = nn.GroupNorm(gn_group, out_channels, affine=not use_film)

        if se_reduction is not None:
            self.se_module = SEModule(out_channels, se_reduction)
        else:
            self.se_module = None

    def forward(self, x, sigma=None, mu=None):
        h = F.relu(self.conv1(x))
        h = self.gn1(h)
        if self.use_film:
            h = h * sigma.view(sigma.size()[0],-1,1) + mu.view(mu.size()[0],-1,1)

        if self.se_module is None:
            h = h + x
        else:
            h = self.se_module(h) + x
        return h

class TransMatrixFilmGenerator(nn.Module):
    def __init__(self, nfeature, num_block, kernel_size=5, channels=256, num_convs=2, 
                 common_neurons=(256,), head_neurons=(128,), dropout_rate=0.1):
        super(TransMatrixFilmGenerator, self).__init__()

        # conv
        self.convs = []
        for i in range(num_convs):
            if i == 0:
                in_ch = 1
            else:
                in_ch = channels
            self.convs.append(nn.Sequential(nn.Conv2d(in_ch, channels, kernel_size, 1, kernel_size//2), 
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(dropout_rate)))
        self.convs = nn.Sequential(*self.convs)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # common
        self.common_layers = []
        in_ch = channels
        for i in range(len(common_neurons)):
            out_ch = common_neurons[i]
            self.common_layers.append(nn.Sequential(nn.Linear(in_ch,out_ch ), 
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(dropout_rate)))
            in_ch = out_ch
        self.common_layers = nn.Sequential(*self.common_layers)

        # head
        self.head_layers = nn.ModuleList()
        for ihd in range(num_block):
            in_ch = common_neurons[-1]

            hd_layer = []
            for i in range(len(head_neurons)):
                out_ch = head_neurons[i]
                hd_layer.append(nn.Sequential(nn.Linear(in_ch, out_ch), 
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout_rate)))
                in_ch = out_ch
            self.head_layers.append(nn.Sequential(*hd_layer))

        # output
        self.output_layers = nn.ModuleList()
        for ihd in range(num_block):
            self.output_layers.append(nn.Linear(head_neurons[-1], nfeature))


    def forward(self, trans_matrix):
        h = trans_matrix.unsqueeze(1)

        # conv
        h = self.convs(h)

        # pooling
        h = self.avg_pool(h)
        h = h.view(h.size()[0], -1)

        # common
        h = self.common_layers(h)

        # head
        hs = []
        for ly in self.head_layers:
            hs.append(ly(h))

        # output
        for i, ly in enumerate(self.output_layers):
            hs[i] = ly(hs[i])

        return hs

class ResNetTransmat1D(nn.Module):
    def __init__(self, nfeature, kernel_size, num_block, input_channels,
                 gn_group, se_reduction, sig_gene=None, mu_gene=None):
        super(ResNetTransmat1D, self).__init__()

        self.first_layer = nn.Sequential(nn.Conv1d(input_channels, nfeature, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False), 
                                         nn.ReLU(inplace=True))

        self.layers = nn.ModuleList()
        use_film = sig_gene is not None and mu_gene is not None
        for i in range(num_block):
            self.layers.append(FilmBlock(nfeature, nfeature, kernel_size, gn_group, se_reduction, use_film))

        self.sig_gene = sig_gene
        self.mu_gene = mu_gene

    def forward(self, x, trans_matrix=None):
        """
        Args:
            x : shape (Batch, C, Length)
            trans_matrix : shape (Batch, Channel, Channel)
        """
        if self.sig_gene is not None and self.mu_gene is not None:
            sigs = self.sig_gene(trans_matrix)
            mus = self.mu_gene(trans_matrix)
        else:
            sigs = [None] * len(self.layers)
            mus = [None] * len(self.layers)

        h = self.first_layer(x)
        for ly, sig, mu in zip(self.layers, sigs, mus):
            h = ly(h, sig, mu)

        return h

