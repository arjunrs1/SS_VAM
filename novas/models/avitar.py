# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import logging
import numpy as np
import torchvision
import speechmetrics

from novas.models.my_conformer import Conformer
from novas.models.vit import ViT
from novas.models.base_av_model import BaseAVModel


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def update_args(args, dic):
    for key, value in dic.items():
        setattr(args, key, value)
    return args


class AViTAr(BaseAVModel):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args

        self.use_visual = args.use_vit or args.use_cnn
        if args.use_rgb or args.use_depth:
            assert self.use_visual

        num_channel = args.use_rgb * 3 + args.use_depth
        if args.use_vit:
            dropout = emb_dropout = 0 if args.no_dropout else args.dropout
            self.vit = ViT(image_size=(192, 576), patch_size=32, num_classes=1000, dim=512, depth=6, heads=16,
                           mlp_dim=2048, dropout=dropout, emb_dropout=emb_dropout, channels=num_channel)
        elif args.use_cnn:
            conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers = list(torchvision.models.resnet18(pretrained=args.pretrained_cnn).children())[1:-2]
            self.cnn = nn.Sequential(conv1, *layers)

        crossmodal_dropout_p = input_dropout_p = feed_forward_dropout_p = attention_dropout_p = conv_dropout_p = 0 if args.no_dropout else args.dropout
        crossmodal_dropout_p = args.crossmodal_dropout_p if args.crossmodal_dropout_p != -1 else crossmodal_dropout_p
        input_dim = 257 if not self.args.encode_wav else args.ngf * (2 ** len(self.args.encoder_ratios.split(',')))
        self.conformer = Conformer(input_dim=input_dim, num_encoder_layers=args.num_encoder_layers,
                                   input_dropout_p=input_dropout_p, feed_forward_dropout_p=feed_forward_dropout_p,
                                   attention_dropout_p=attention_dropout_p, conv_dropout_p=conv_dropout_p,
                                   use_crossmodal_layer=self.use_visual,
                                   conv_kernel_size=args.conv_kernel_size, decode_wav=args.decode_wav,
                                   encode_wav=args.encode_wav, encoder_ratios=args.encoder_ratios,
                                   decoder_ratios=args.decoder_ratios, encoder_residual_layers=args.encoder_residual_layers,
                                   decoder_residual_layers=args.decoder_residual_layers,
                                   ngf=args.ngf, use_visual_pe=self.args.use_visual_pe, crossmodal_dropout_p=crossmodal_dropout_p,
                                   )

        non_learnable_feat_dim = sum([args.use_tgt_pose, args.use_speaker_pose]) * 2 + \
                                 4 * args.use_speaker_bboxes + args.use_tgt_rotation + 64 * args.use_depth_mini + \
                                 4 * args.use_tgt_bboxes
        if args.encode_sincos:
            non_learnable_feat_dim += sum([args.use_tgt_pose, args.use_speaker_pose, args.use_tgt_rotation])

        input_dim = 512 * (args.use_rgb + args.use_tgt_rgb) + 864 * args.use_tgt_rgb_pano + non_learnable_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance', 'delay', 'lr_ratio', 'lr_ratio_peak', 'l2_distance']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def get_visual_feat(self, batch):
        visual_input = []
        if self.args.use_depth:
            visual_input.append(batch['depth'])
        if self.args.use_rgb:
            visual_input.append(batch['src_rgb'])
        visual_input = torch.cat(visual_input, dim=1)

        if self.args.use_vit:
            img_feat = self.vit(visual_input)
        elif self.args.use_cnn:
            src_feat = self.cnn(batch['src_rgb'])
            src_feat = src_feat.reshape(src_feat.size(0), src_feat.size(1), -1).permute(0, 2, 1)
            tgt_feat = self.cnn(batch['tgt_rgb'])
            tgt_feat = src_feat.reshape(tgt_feat.size(0), tgt_feat.size(1), -1).permute(0, 2, 1)
            img_feat = torch.concat([src_feat, tgt_feat], dim=1)
        else:
            img_feat = None

        return img_feat

    def print(self, string):
        if self.global_rank == 0:
            logging.info(string)

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.decode_wav:
            if self.args.encode_wav:
                src = batch['src_wav']
                tgt = batch['tgt_wav']
            else:
                src = self.wav2spec(batch['src_wav'])[..., 0].permute(0, 2, 1)
                tgt = torch.nn.functional.pad(batch['tgt_wav'], (0, 128))
        else:
            src = self.wav2spec(batch['src_wav'])[..., 0].permute(0, 2, 1)
            tgt = self.wav2spec(batch['tgt_wav'])[..., 0].permute(0, 2, 1)

        img_feat = self.get_visual_feat(batch) if self.use_visual else None
        non_learnable_features = []
        if self.args.use_tgt_pose:
            non_learnable_features.append(batch['tgt_pose'])
        if self.args.use_tgt_rotation:
            non_learnable_features.append(batch['tgt_rotation'])
        if self.args.use_speaker_pose:
            non_learnable_features.append(batch['speaker_pose'])
        if self.args.use_speaker_bboxes:
            non_learnable_features.append(batch['bboxes'])
        if self.args.use_tgt_bboxes:
            non_learnable_features.append(batch['tgt_bboxes'])
        non_learnable_features = torch.concat(non_learnable_features, dim=1).unsqueeze(1)
        non_learnable_features = non_learnable_features.expand((non_learnable_features.shape[0], img_feat.shape[1],
                                                                non_learnable_features.shape[2]))

        features = self.fusion(torch.concat([non_learnable_features, img_feat], dim=2))
        pred = self.conformer(src, img_feat=features)

        return {'pred': pred, 'tgt': tgt}
