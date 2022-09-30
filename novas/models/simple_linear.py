# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import speechmetrics

from novas.models.base_av_model import BaseAVModel
from novas.models.binaural_nets import HyperConvWavenet, WaveoutBlock


class SimpleLinear(BaseAVModel):
    def __init__(self, args):
        super(SimpleLinear, self).__init__(args)
        self.args = args

        self.dilation = 1
        self.kernel_size = args.kernel_size
        self.linear_layer = nn.Conv1d(2, 2, kernel_size=self.kernel_size, dilation=self.dilation)
        self.linear_layer.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def forward(self, batch):
        padding = self.dilation * (self.kernel_size - 1)
        input_wav = F.pad(batch['src_wav'], [padding, 0])
        output = self.linear_layer(input_wav)

        return output

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        pred_wav = self.forward(batch)

        return {'pred': pred_wav, 'tgt': batch['tgt_wav']}
