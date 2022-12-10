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
from novas.models.binaural_nets import HyperConvWavenet, WaveoutBlock, Warpnet
from novas.models.vida import create_conv
from novas.models.melgan_encoder import MelGANEncoder

DEREVERBERATOR = None

def update_args(args, dic):
    for key, value in dic.items():
        setattr(args, key, value)
    return args


def load_dereverberator(device, use_vida=False):
    if use_vida:
        from novas.vam.models.vida import VIDA
        from trainer import parser
        pretrained_weights = 'data/pretrained-models/dereverb/vida.pth'
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        default_args = parser.parse_args("")
        update_args(default_args, {'use_rgb': False, 'use_depth': False, 'log1p': True, 'decode_wav': True, 'dereverb': True})
        dereverberator = VIDA(default_args).to(device=device)
        dereverberator.load_weights(state_dict)
        dereverberator.eval()
    else:
        from trainer import parser
        pretrained_weights = 'data/pretrained-models/dereverb/best_val.ckpt'
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        default_args = parser.parse_args("")
        update_args(default_args, 
                vars(state_dict['hparams']))
        dereverberator = WaveNet(default_args).to(device=device)
        dereverberator.load_state_dict(state_dict['state_dict'])
        dereverberator.eval()

    return dereverberator


class WaveNet(BaseAVModel):
    def __init__(self, args):
        super(WaveNet, self).__init__(args)
        self.args = args

        non_learnable_feat_dim = sum([args.use_tgt_pose, args.use_speaker_pose, args.use_speaker_pose_wrt_tgt]) * 2 + \
                                 4 * args.use_speaker_bboxes + args.use_tgt_rotation + 64 * args.use_depth_mini + \
                                 4 * args.use_tgt_bboxes + 2 * args.use_src_location + 2 * args.use_tgt_location

        if args.encode_sincos:
            non_learnable_feat_dim += sum([args.use_tgt_pose, args.use_speaker_pose, args.use_tgt_rotation, args.use_speaker_pose_wrt_tgt])
        view_dim = 256 if args.use_rgb or args.mlp_fusion else non_learnable_feat_dim
        view_dim = 256 if view_dim == 0 else view_dim
        self.view_dim = view_dim
        wavenet_blocks = args.wavenet_blocks
        layers_per_block = args.layers_per_block
        wavenet_channels = 64
        if args.use_rgb or args.use_tgt_rgb or args.use_tgt_rgb_pano:
            layers = list(torchvision.models.resnet18(pretrained=True).children())[0:-2]
            if args.use_depth:
                conv2d = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.visual_encoder = nn.Sequential(conv2d, *layers[1:])
            else:
                self.visual_encoder = nn.Sequential(*layers)
            self.conv1x1 = create_conv(512, 8, 1, 0)

            input_dim = 512 * (args.use_rgb + args.use_tgt_rgb) + 864 * args.use_tgt_rgb_pano + non_learnable_feat_dim
            input_dim = 864 #480
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        elif args.use_depth:
            assert not (self.args.use_rgb or self.args.use_tgt_rgb or self.args.use_tgt_depth)
            conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers = list(torchvision.models.resnet18(pretrained=True).children())[0:-2]
            self.visual_encoder = nn.Sequential(conv1, *layers[1:])
            self.conv1x1 = create_conv(512, 8, 1, 0)

            input_dim = 512 + non_learnable_feat_dim
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        elif args.mlp_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(non_learnable_feat_dim, 512),
                nn.ReLU(),
                nn.Linear(512, view_dim)
            )
        else:
            self.fusion = torch.nn.Identity()

        if args.use_warper:
            self.warper = Warpnet(view_dim=view_dim)

        if args.scale_mag:
            self.mag_scaler = nn.Sequential(
                nn.Linear(view_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )

        if args.wavenet_encoder == 'hyperconv':
            self.input = nn.Conv1d(1, wavenet_channels, kernel_size=1)
            self.input.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))
            self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels, wavenet_blocks, layers_per_block,
                                                      remove_hyperconv=args.remove_hyperconv,
                                                      conv_len=3 if args.acausal else 2,
                                                      acausal=args.acausal)
            self.output_net = nn.ModuleList([WaveoutBlock(wavenet_channels)
                                            for _ in range(wavenet_blocks*layers_per_block)])
        elif args.wavenet_encoder == 'melgan':
            self.melgan_encoder = MelGANEncoder(view_dim, wavenet_channels, wavenet_blocks, layers_per_block)

        if args.use_spatial_classification_loss:
            self.audio_feat_net = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
            )
            self.audio_visual_fusion = nn.Linear(1024, 512)
            self.spatial_classifier = nn.Linear(512, 1)

        if args.use_viewpoint_contrastive_loss:
            self.audio_feat_net = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
            )
            self.triplet_loss = nn.TripletMarginLoss(args.viewpoint_triplet_margin)

        self.first_val = True
        self.configure_loss()
        if args.use_librispeech and args.eval_clean:
            self.metrics = []
        else:
            self.metrics = ['stft_distance', 'mag_distance', 'l2_distance']
        self.speech_metric_names = {'stoi', 'sisdr', 'bsseval', 'mosnet'}.intersection(self.metrics)
        self.speech_metrics = speechmetrics.load(self.speech_metric_names, window=None)


    def compute_visual_feat(self, rgb):
        feat = self.visual_encoder(rgb)
        feat = self.conv1x1(feat).reshape(feat.shape[0], -1)
        return feat

    def forward(self, batch):
        features = []
        if self.args.use_rgb:
            if self.args.use_depth: 
                rgbd = torch.cat((batch['src_depth'], batch['src_rgb']),1)
                features.append(self.compute_visual_feat(rgbd))
            else:
                features.append(self.compute_visual_feat(batch['src_rgb']))
        else:
            if self.args.use_depth:
                features.append(self.compute_visual_feat(batch['src_depth']))
        if self.args.use_depth_mini:
            features.append(batch['depth_mini'])
        if self.args.use_tgt_rgb:
            features.append(self.compute_visual_feat(batch['tgt_rgb']))
        if self.args.use_tgt_rgb_pano:
            features.append(self.compute_visual_feat(batch['tgt_rgb_pano']))
        if self.args.use_tgt_pose:
            features.append(batch['tgt_pose'])
        if self.args.use_tgt_rotation:
            features.append(batch['tgt_rotation'])
        if self.args.use_speaker_pose:
            features.append(batch['speaker_pose'])
        if self.args.use_speaker_pose_wrt_tgt:
            features.append(batch['speaker_pose_wrt_tgt'])
        if self.args.use_speaker_bboxes:
            features.append(batch['bboxes'])
        if self.args.use_tgt_bboxes:
            features.append(batch['tgt_bboxes'])
        if self.args.use_src_location:
            features.append(batch['src_location'])
        if self.args.use_tgt_location:
            features.append(batch['tgt_location'])

        if (self.args.use_rgb == False) and (self.args.use_depth == False):
            if len(batch['src_wav'].shape) != 2:
                batch['src_wav'] = torch.unsqueeze(batch['src_wav'], 0)
            features = torch.ones((batch['src_wav'].shape[0],self.view_dim)).to(self.device)
            features = self.fusion(features)
        else:
            features = self.fusion(torch.concat(features, dim=1))

        x = batch['src_wav']
        if self.args.use_warper:
            x = self.warper(x, features.unsqueeze(-1))

        # B x 2 X T
        scale = self.mag_scaler(features) if self.args.scale_mag else torch.tensor([1.0]).to(self.device)
        x = x * scale.unsqueeze(-1)

        if self.args.wavenet_encoder == 'identity':
            x = [x, x]
        elif self.args.wavenet_encoder == 'hyperconv':

            x = torch.unsqueeze(x,1)

            x = self.input(x)
            output, skips = self.hyperconv_wavenet(x, features.unsqueeze(-1))
            # collect output and skips after each layer, "output": x[0], "intermediate": x[1:]
            x = []
            if self.args.use_last_output:
                x = [self.output_net[-1](output), self.output_net[-1](output)]
            else:
                for k in range(len(skips), 0, -1):
                    y = torch.mean(torch.stack(skips[:k], dim=0), dim=0) if not self.args.no_stacking else skips[k-1]
                    y = self.output_net[k-1](y)
                    x += [y]
        elif self.args.wavenet_encoder == 'melgan':
            x = self.melgan_encoder(x, features.unsqueeze(-1))

        return x[0], x[1:], scale

    def audio_synthesis(self, batch, batch_idx, phase=None):
        for key, value in batch.items():
            batch[key] = value.to(device=self.device, dtype=torch.float)

        if self.args.randomize_input:
            batch['src_wav'] = torch.roll(batch['src_wav'], shifts=1, dims=0)

        #if dereverberation flaggged, perform dereverberation
        if self.args.use_frozen_dereverb:
            dereverb_sspeech(batch, self.args.encode_wav, self.args.randomize_input, self.device, use_vida=self.args.use_vida)
        batch['original_src_wav'] = batch['src_wav']

        if self.args.convolve_random_rir:
            convolve_random_rir(batch)

        if self.args.use_audio_da:
            apply_audio_data_augmentation(self.args, batch, phase)
        
        pred_wav, intermediate_wav, scale = self.forward(batch)

        if self.args.subtract_mean:
            pred_wav -= torch.mean(pred_wav, dim=-1, keepdim=True)

        
        return {'pred': pred_wav, 'tgt': batch['tgt_wav'], 'intermediate': intermediate_wav, 'batch': batch,
                'scale': scale}

    def compute_audio_feat(self, batch):
        window = torch.hamming_window(400, device=self.device)
        input_wav = batch['src_wav'].reshape(-1, *batch['src_wav'].shape[2:])
        audio_spec = torch.stft(input_wav, n_fft=512, hop_length=160, win_length=400, window=window,
                                pad_mode='constant', center=True, return_complex=True).abs()
        audio_spec = audio_spec.reshape(-1, 2, *audio_spec.shape[1:])
        audio_feat = self.audio_feat_net(audio_spec).squeeze(-1).squeeze(-1)

        return audio_feat

    def compute_loss(self, outputs):
        stats = super().compute_loss(outputs)
        batch = outputs['batch']
        pred, tgt = outputs['pred'], outputs['tgt']

        if self.args.use_intermediate_loss:
            tgt = tgt.reshape(-1, tgt.shape[-1])
            intermediate = outputs['intermediate']
            for pred in intermediate:
                pred = pred.reshape(-1, pred.shape[-1])
                loss = self.loss(pred, tgt)
                stats['loss'] = stats['loss'] + loss

        if self.args.use_spatial_classification_loss:
            audio_feat = self.compute_audio_feat(batch)

            rgb = batch['src_rgb']
            new_rgb = []
            flipped = np.zeros([rgb.shape[0]])
            for i in range(rgb.shape[0]):
                if np.random.random() > 0.5:
                    new_rgb.append(torch.flip(rgb[i], dims=[2]))
                    flipped[i] = 1
                else:
                    new_rgb.append(rgb[i])
                    flipped[i] = 0
            new_rgb = torch.stack(new_rgb, dim=0)
            batch['src_rgb'] = new_rgb

            # if flipped:
            #     import matplotlib.pyplot as plt
            #     plt.imsave('flipped.png', rgb[0].permute(1, 2, 0).cpu().numpy())
            #     exit()

            visual_feat = self.compute_visual_feat(new_rgb)
            fused_feature = self.audio_visual_fusion(torch.concat([visual_feat, audio_feat], dim=1))
            binary_pred = self.spatial_classifier(F.relu(fused_feature))

            tgt_label = torch.tensor(flipped).unsqueeze(-1).float().to(self.device)
            binary_loss = F.binary_cross_entropy_with_logits(binary_pred, tgt_label)
            stats['binary_loss'] = binary_loss
            stats['binary_accuracy'] = ((binary_pred > 0.5) == tgt_label).float()
            stats['loss'] = stats['loss'] + binary_loss * self.args.spatial_classification_loss_weight

        if self.args.use_viewpoint_contrastive_loss:
            audio_feat = self.compute_audio_feat(batch)
            pos_visual_feat = self.compute_visual_feat(batch['src_rgb'])
            neg_visual_feat = self.compute_visual_feat(batch['tgt_rgb'])

            triplet_loss = self.triplet_loss(audio_feat, pos_visual_feat, neg_visual_feat)
            stats['triplet_loss'] = triplet_loss
            stats['loss'] = stats['loss'] + triplet_loss * self.args.viewpoint_contrastive_loss_weight

        if self.args.use_scale_loss:
            if self.args.use_peak_scale:
                src_max = batch['src_wav'].abs().max(dim=-1)[0]
                tgt_max = batch['tgt_wav'].abs().max(dim=-1)[0]
            elif self.args.use_rms_scale:
                src_max = batch['src_wav'].pow(2).mean(dim=-1).sqrt()
                tgt_max = batch['tgt_wav'].pow(2).mean(dim=-1).sqrt()
            else:
                raise ValueError

            gt_scale = torch.clip(tgt_max / (src_max + 1E-5), 0, 100)
            scale_loss = F.mse_loss(gt_scale, outputs['scale'])
            stats['scale_loss'] = scale_loss
            stats['loss'] = stats['loss'] + scale_loss * self.args.scale_loss_weight

        return stats

def dereverb_sspeech(batch, encode_wav, randomize_input, device, use_vida=False):
    with torch.no_grad():
        global DEREVERBERATOR
        if DEREVERBERATOR is None:
            DEREVERBERATOR = load_dereverberator(device, use_vida)
        
        if randomize_input:
            original_rgb = batch['src_rgb']
            original_depth = batch['src_depth']
            batch['src_rgb'] = torch.roll(batch['src_rgb'], shifts=1, dims=0)
            batch['src_depth'] = torch.roll(batch['src_depth'], shifts=1, dims=0)
        if use_vida:
            pred_clean = DEREVERBERATOR.dereverb(batch, 0)['pred']
        else:
            pred_clean = DEREVERBERATOR.audio_synthesis(batch, 0)['pred']
        if randomize_input:
            #restore original rgb tensor for use in reverberation model later
            batch['src_rgb'] = original_rgb
            batch['src_depth'] = original_depth
        if encode_wav:
            batch['src_wav'] = torch.squeeze(pred_clean[:, :batch['tgt_wav'].shape[1]])
        else:
            batch['src_wav'] = pred_clean

def convolve_random_rir(batch):
    rirs = torch.flip(batch['rir'], (1,)).unsqueeze(1)
    outputs = []
    for i in range(batch['src_wav'].shape[0]):
        rir = rirs[i: i + 1]
        audio = batch['src_wav'].unsqueeze(1)[i:i + 1]
        outputs.append(torch.nn.functional.conv1d(audio, rir, padding=rir.shape[-1] - 1))
    outputs = torch.cat(outputs, dim=0).squeeze(1)
    batch['src_wav'] = outputs[:, :batch['tgt_wav'].shape[1]].detach()
    batch['src_wav'] /= batch['src_wav'].abs().max(dim=1, keepdim=True)[0]
    batch['src_wav'] = torch.nan_to_num(batch['src_wav'])
    return batch

def apply_audio_data_augmentation(args, batch, phase):
    from torch_audiomentations import Compose, PolarityInversion, AddColoredNoise

    transforms = []
    transforms.append(AddColoredNoise(p=1, min_snr_in_db=args.min_snr, max_snr_in_db=args.max_snr))
    transforms.append(PolarityInversion(p=0.5))
    
    # Initialize augmentation callable
    if len(transforms) != 0:
        apply_augmentation = Compose(transforms=transforms)
        prev_shape = batch['src_wav'].shape
        batch['src_wav'] = apply_augmentation(batch['src_wav'].unsqueeze(1), sample_rate=16000).squeeze(1)
        assert batch['src_wav'].shape == prev_shape, (batch['src_wav'].shape, prev_shape)