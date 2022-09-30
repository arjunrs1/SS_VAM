import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np

from novas.models.wav_decoder import weights_init, WNConv1d, WNConvTranspose1d


class WNConditionalCon1d(nn.Module):
    def __init__(self, input_size, ch_in, ch_out, kernel_size, dilation=1):
        '''
        HyperConv implements a temporal convolution that has different convolution weights for each time step.
        :param input_size: (int) dimension of the weight generating input variable
        :param ch_in: (int) number of input channels of the temporal convolution
        :param ch_out: (int) number of output channels of the temporal convolution
        :param kernel_size: (int) kernel size of the temporal convolution
        :param dilation: (int) dilation of the temporal convolution
        '''
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.w_f = WNConv1d(ch_in, ch_out, kernel_size=kernel_size, dilation=dilation)
        self.w_g = WNConv1d(ch_in, ch_out, kernel_size=kernel_size, dilation=dilation)
        self.v_f = WNConv1d(input_size, ch_out, kernel_size=1)
        self.v_g = WNConv1d(input_size, ch_out, kernel_size=1)

        for net in [self.w_f, self.w_g, self.v_g, self.v_f]:
            net.weight.data.zero_()
            net.bias.data.uniform_(-np.sqrt(6.0/(self.ch_in*self.kernel_size)),
                                   np.sqrt(6.0/(self.ch_in*self.kernel_size)))

    def forward(self, x, z):
        '''
        :param x: the input signal as a B x ch_in x T tensor
        :param z: the weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: a B x ch_out x T tensor as the result of the hyper-convolution
        '''
        B, S = x.shape[0], x.shape[-1]
        assert x.shape[-1] % z.shape[-1] == 0

        # global conditioning
        y = torch.tanh(self.w_f(x) + self.v_f(z).expand((B, self.ch_out, S))) \
            * torch.sigmoid(self.w_g(x) + self.v_g(z).expand((B, self.ch_out, S)))

        return y


class ConditionalResnetBlock(nn.Module):
    def __init__(self, dim, z_dim, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
        )
        self.conditional_conv = WNConditionalCon1d(z_dim, dim, dim, kernel_size=1)
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x, z):
        return self.shortcut(x) + self.conditional_conv(self.block(x), z)


class WaveoutBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2),
            WNConv1d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            WNConv1d(channels, 2, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)


class MelGANEncoder(nn.Module):
    def __init__(self, z_dim, channels=64, blocks=3, layers_per_block=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad1d(3),
            WNConv1d(2, channels, kernel_size=7, padding=0),
        )

        self.rectv_field = 1

        layers = []
        for b in range(blocks):
            for l in range(layers_per_block):
                dilation = 3 ** l
                layers += [
                    ConditionalResnetBlock(channels, z_dim, dilation=3 ** l)
                ]
                self.rectv_field += 3 * dilation
        self.layers = nn.ModuleList(layers)

        self.decoder = nn.ModuleList([WaveoutBlock(channels) for _ in range(blocks * layers_per_block)])
        self.apply(weights_init)

    def forward(self, x, z):
        '''
        :param x: input signal as a B x channels x T tensor
        :param z: weight-generating input as a B x z_dim z K tensor (K = T / 400)
        :return: x: output signal as a B x channels x T tensor
                 skips: skip signal for each layer as a list of B x channels x T tensors
        '''
        outputs = []
        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, z)
            outputs += [self.decoder[i](x)]
        return outputs[::-1]

    def receptive_field(self):
        return self.rectv_field
