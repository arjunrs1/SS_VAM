# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torch.nn.functional as F
import speechmetrics

from novas.models.base_av_model import BaseAVModel
from novas.models.vida import unet_conv, unet_upconv, VisualNet, create_conv


class UNET(BaseAVModel):
    def __init__(self, args):
        super(UNET, self).__init__(args)
        self.args = args
        ngf = 64
        input_nc = output_nc = 2 * args.num_channel if args.predict_mask else args.num_channel
        self.use_visual = args.use_depth + args.use_rgb + args.use_tgt_rgb + args.use_tgt_rgb_pano
        if self.use_visual:
            self.visual_encoder = VisualNet(torchvision.models.resnet18(pretrained=True),
                args.use_depth + (args.use_rgb or args.use_tgt_rgb or args.use_tgt_rgb_pano) * 3)
            self.conv1x1 = create_conv(512, 8, 1, 0)

        num_intermediate_feature = 512
        if args.dataset == 'replay':
            num_intermediate_feature += 672 * 2 if self.use_visual else 0
        else:
            num_intermediate_feature += 512 * (args.use_rgb + args.use_tgt_rgb)
            num_intermediate_feature += 864 * self.args.use_tgt_rgb_pano
            num_intermediate_feature += 2 * self.args.use_tgt_pose
            num_intermediate_feature += 2 * self.args.use_speaker_pose
            num_intermediate_feature += 4 * self.args.use_tgt_bboxes
            if args.use_tgt_rotation:
                assert args.encode_sincos
                num_intermediate_feature += sum([args.use_tgt_pose, args.use_speaker_pose])

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(num_intermediate_feature, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True, use_tanh=args.predict_mask)

        self.first_val = True
        self.configure_loss()
        self.metrics = ['stft_distance', 'mag_distance']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)

    def forward(self, audio, src_visual, tgt_visual, non_learnable_feat):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        upconv_feature_input = audio_conv5feature
        for visual in [src_visual, tgt_visual]:
            if visual is None:
                continue
            visual_feat = self.conv1x1(self.visual_encoder(visual))
            visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
            visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2],
                                             audio_conv5feature.shape[-1])  # tile visual feature
            upconv_feature_input = torch.cat((visual_feat, upconv_feature_input), dim=1)

        if non_learnable_feat is not None:
            non_learnable_feat = non_learnable_feat.view(non_learnable_feat.shape[0], -1, 1, 1)
            non_learnable_feat = non_learnable_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])
            upconv_feature_input = torch.cat((non_learnable_feat, upconv_feature_input), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(upconv_feature_input)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))

        return prediction

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)
        if self.args.audio_len == 16000:
            batch['src_wav'] = F.pad(batch['src_wav'], (0, 40960 - 16000))
            batch['tgt_wav'] = F.pad(batch['tgt_wav'], (0, 40960 - 16000))

        src_spec = self.wav2spec(batch['src_wav']).permute(0, 1, 4, 2, 3)
        tgt_spec = self.wav2spec(batch['tgt_wav']).permute(0, 1, 4, 2, 3)
        if not self.args.predict_mask:
            src_spec = src_spec[:, :, :1, :, :]
            tgt_spec = tgt_spec[:, :, :1, :, :]

        src_visual = batch['src_rgb'] if self.args.use_rgb else None
        tgt_visual = batch['tgt_rgb'] if self.args.use_tgt_rgb else None

        non_learnable_features = []
        if self.args.use_tgt_pose:
            non_learnable_features.append(batch['tgt_pose'])
        if self.args.use_speaker_pose:
            non_learnable_features.append(batch['speaker_pose'])
        if self.args.use_speaker_bboxes:
            non_learnable_features.append(batch['tgt_bboxes'])
        non_learnable_features = torch.concat(non_learnable_features, dim=1)

        s, f = src_spec.shape[1:3]
        # predict both channels together instead of reshaping
        input_spec = src_spec.reshape((src_spec.shape[0], s * f, src_spec.shape[3], src_spec.shape[4]))
        pred = self.forward(input_spec, src_visual, tgt_visual, non_learnable_features)
        pred = pred.reshape((pred.shape[0], s, f, pred.shape[2], pred.shape[3]))

        src_spec_copy = src_spec.clone().detach()
        if self.args.predict_mask:
            src_spec_copy[:, :, :, :-1, :-1] *= pred * self.args.mask_scale
        else:
            src_spec_copy[:, :, :, :-1, :-1] = pred[:, :, :, :, :]
        pred_spec = src_spec_copy
        assert pred_spec.shape == tgt_spec.shape, (pred_spec.shape, tgt_spec.shape)

        return {'pred': pred_spec, 'tgt': tgt_spec}
