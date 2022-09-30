# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import itertools
import torch.nn.functional as F

from novas.models.wavenet import WaveNet
from novas.models.generative_model import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, \
    generator_loss, discriminator_loss, mel_spectrogram


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class GenerativeWaveNet(WaveNet):
    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters(args)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def training_step(self, batch, batch_idx):
        optim_d, optim_g = self.optimizers()

        output = self.audio_synthesis(batch, batch_idx, phase='train')
        pred, tgt = output['pred'], output['tgt']

        stats = {}
        loss_disc_all = torch.tensor([.0]).to(self.device)
        loss_mel = torch.tensor([.0]).to(self.device)
        for y_g_hat, y in zip([pred[:, :1], pred[:, 1:2]], [tgt[:, :1], tgt[:, 1:2]]):
            y_mel = mel_spectrogram(y.squeeze(1), 1024, 80, 160000, 256, 1024, 0, None)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 1024, 80, 160000, 256, 1024, 0, None)

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all += loss_disc_s + loss_disc_f

            # L1 Mel-Spectrogram Loss
            loss_mel += F.l1_loss(y_mel, y_g_hat_mel) * self.args.stft_loss_weight \
                if not self.args.remove_mel_loss else 0

        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        optim_d.step()

        if self.args.multires_stft:
            ret = self.loss(pred, tgt)
            loss_multires_stft = (ret[0] + ret[1]) * self.args.stft_loss_weight
            stats['multires_stft'] = loss_multires_stft.detach()
        else:
            loss_multires_stft = 0

        if self.args.rel_mag:
            rel_mag = stats['rel_mag'] = self.loss(pred, tgt).detach()
        else:
            rel_mag = 0

        loss_gen_all = torch.tensor([.0]).to(self.device)
        for y_g_hat, y in zip([pred[:, :1], pred[:, 1:2]], [tgt[:, :1], tgt[:, 1:2]]):
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) if not self.args.remove_fm_loss else 0
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) if not self.args.remove_fm_loss else 0
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all += loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_multires_stft + rel_mag

        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
        optim_g.step()

        stats['generator'] = loss_gen_all.detach()
        stats['discriminator'] = loss_disc_all.detach()

        return stats

    @property
    def automatic_optimization(self) -> bool:
        return False

    def configure_optimizers(self):
        generator_params = set(self.parameters()) - set(self.msd.parameters()) - set(self.mpd.parameters())
        optim_g = torch.optim.AdamW(generator_params, self.args.generator_lr, betas=[0.8, 0.99])
        optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                    self.args.discriminator_lr, betas=[0.8, 0.99])
        return optim_d, optim_g

