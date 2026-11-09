#!/usr/bin/env/python3
"""
Recipe for training MetricGAN-U (Unsupervised) with the Voicebank dataset.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Szu-Wei Fu 2021/09
"""

import os
import sys
import shutil
import torch
import torchaudio
import speechbrain as sb
import numpy as np
import json
import pickle
import requests
import time
import tqdm
import string
import logging
import matplotlib.pyplot as plt
import copy

from urllib.parse import urlparse, urljoin
from srmrpy import srmr
from enum import Enum, auto
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats as SpeechBrainMetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler


from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

import torch
import torchaudio
import torchvision
from PIL import Image

import novas
from novas.models.base_av_model import load_rt60_estimator, measure_rt60_wrapper
from novas.datasets.avspeech_dataset import AVSpeechDataset
dataset = AVSpeechDataset


### For DNSMSOS
# URL for the web service
SCORING_URI = "https://dnsmos-4.azurewebsites.net/score"
# If the service is authenticated, set the key or token
AUTH_KEY = os.environ.get("DNSMOS_AUTH_KEY", "")
# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Basic {AUTH_KEY }"

class MetricStats(SpeechBrainMetricStats):
    """SpeechBrain metric accumulator with per-example image arguments."""

    def append(self, ids, *args, **kwargs):
        rgb = kwargs.pop("rgb", None)
        if rgb is None:
            return super().append(ids, *args, **kwargs)
        self.ids.extend(ids)
        predict, target = kwargs["predict"], kwargs["target"]
        lengths = kwargs.get("lengths")
        if self.batch_eval:
            scores = self.metric(predict=predict, target=target, lengths=lengths, rgb=rgb)
            self.scores.extend(torch.as_tensor(scores).detach().cpu().reshape(-1).tolist())
            return
        for i, (pred, tgt, image) in enumerate(zip(predict, target, rgb)):
            end = round(float(lengths[i]) * predict.size(1)) if lengths is not None else predict.size(1)
            self.scores.append(self.metric(pred[:end].detach().cpu(), tgt[:end].detach().cpu(), image.detach().cpu()))


def update_args(args, dic, lr=None):
    for key, value in dic.items():
        if key == "lr" and lr is not None:
            value = lr
        if key == "use_frozen_dereverb":
            value = False
        setattr(args, key, value)
    return args


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def pesq_eval(predict, target):
    """Normalized PESQ (to 0-1)"""
    return (
        pesq(fs=16000, ref=target.numpy(), deg=predict.numpy(), mode="wb") + 0.5
    ) / 5


def srmrpy_eval(predict, target):
    """ Note target_wav is not used in the srmr function !!!
        Normalize the score to 0~1 for training.
    """

    try:
        return float(
        sigmoid(
            0.1
            * srmr(
                predict.numpy(),
                fs=16000,
                n_cochlear_filters=23,
                low_freq=125,
                min_cf=4,
                max_cf=128,
                fast=True,
                norm=False,
            )[0]
            )
        )
    except IndexError:
        return 0


def srmrpy_eval_valid(predict, target):
    """ Note target_wav is not used in the srmr function !!!
        Show the unnormalized score for valid and test set.
    """

    try:
        return float(
            srmr(
            predict.numpy(),
            fs=16000,
            n_cochlear_filters=23,
            low_freq=125,
            min_cf=4,
            max_cf=128,
            fast=True,
            norm=False,
            )[0]
        )
    except IndexError:
        return 0


def dnsmos_eval(predict, target):
    """ Note target_wav is not used in the dnsmos function !!!
        Normalize the score to 0~1 for training.
    """
    pred_wav = predict

    pred_wav = pred_wav.numpy()
    pred_wav = pred_wav / max(abs(pred_wav))
    data = {"data": pred_wav.tolist()}

    input_data = json.dumps(data)
    while True:
        try:
            u = urlparse(SCORING_URI)
            resp = requests.post(
                urljoin("https://" + u.netloc, "score"),
                data=input_data,
                headers=headers,
            )
            score_dict = resp.json()
            score = float(
                sigmoid(score_dict["mos"])
            )  # normalize the score to 0~1
            break
        except Exception as e:  # sometimes, access the dnsmos server too ofen may disable the service.
            print(e)
            time.sleep(10)  # wait for 10 secs
    return score


def dnsmos_eval_valid(predict, target):
    """ Note target_wav is not used in the dnsmos function !!!
        Show the unnormalized score for valid and test set.
    """
    pred_wav = predict

    pred_wav = pred_wav.numpy()
    pred_wav = pred_wav / max(abs(pred_wav))
    data = {"data": pred_wav.tolist()}
    input_data = json.dumps(data)
    while True:
        try:
            u = urlparse(SCORING_URI)
            resp = requests.post(
                urljoin("https://" + u.netloc, "score"),
                data=input_data,
                headers=headers,
            )
            score_dict = resp.json()
            score = float(score_dict["mos"])
            break
        except Exception as e:  # sometimes, access the dnsmos server too ofen may disable the service.
            print(e)
            time.sleep(10)  # wait for 10 secs
    return score

def mosnet_eval(predict, target):
    """ Note target_wav is not used in the srmr function !!!
        Normalize the score to 0~1 for training.
    """
    #TO DO: Implement this with mosnet function from speechmetrics
    try:
        pass
    except IndexError:
        return 0

def construct_batch(pred, tgt, rgb):
    batch = {}
    if len(pred.shape) != 2:
        pred = pred.unsqueeze(0)
    if len(tgt.shape) != 2:
        tgt = tgt.unsqueeze(0)
    if len(rgb.shape) != 4:
        rgb = rgb.unsqueeze(0)

    batch['src_wav'] = pred
    batch['tgt_wav'] = tgt
    if rgb.shape[1] == 4:
        batch['src_depth'] = rgb[:,-1,:,:].unsqueeze(1)
        batch['src_rgb'] = rgb[:,:-1,:,:]
    else:
        batch['src_rgb'] = rgb

    return batch

class SubStage(Enum):
    """For keeping track of training stage progress"""

    GENERATOR = auto()
    CURRENT = auto()
    HISTORICAL = auto()


class MetricGanBrain(sb.Brain):
    def __init__(self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        ):

        super(MetricGanBrain, self).__init__(
        modules=modules,
        opt_class=opt_class,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
        )

        if "soundspaces_speech" in hparams['data_folder']:
            self.use_soundspaces = True
        else:
            self.use_soundspaces = False

        self.device = run_opts['device']

        self.vis_reverb_tgt, self.vis_reverb_opt = self.load_vis_reverberator(self.device, lr=float(hparams['dynamic_rt60_lr']),
            use_devb=hparams['use_devb'], use_soundspaces=self.use_soundspaces)

        self.no_vis_reverb_tgt, self.no_vis_reverb_opt = self.load_no_vis_reverberator(self.device, lr=float(hparams['dynamic_rt60_lr']),
            use_devb=hparams['use_devb'], use_soundspaces=self.use_soundspaces)

        self.vis_reverb_copy, _ = self.load_vis_reverberator(self.device, lr=float(hparams['dynamic_rt60_lr']),
            use_devb=hparams['use_devb'], use_soundspaces=self.use_soundspaces)

        self.no_vis_reverb_copy, _ = self.load_no_vis_reverberator(self.device, lr=float(hparams['dynamic_rt60_lr']),
            use_devb=hparams['use_devb'], use_soundspaces=self.use_soundspaces)

        self.rt60_estimator = load_rt60_estimator(self.device)


    def load_history(self, generate_dataset=False):
        if generate_dataset:
            return

        try:
            if os.path.isfile(self.hparams.historical_file):
                with open(self.hparams.historical_file, "rb") as fp:  # Unpickling
                        self.historical_set = pickle.load(fp)
            else:
                print("no historical file found")
                self.historical_set = {}
        except:
            print("unable to open historical file...creating new file")
            self.historical_set = {}

    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        spec = spectral_magnitude(feats, power=0.5)
        return spec

    def compute_forward(self, batch, stage):
        "Given an input batch computes the enhanced signal"
        batch = batch.to(self.device)

        if self.sub_stage == SubStage.HISTORICAL:
            predict_wav, lens = batch.enh_sig
            return predict_wav
        else:
            noisy_wav, lens = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)

            mask = self.modules.generator(noisy_spec, lengths=lens)
            mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
            predict_spec = torch.mul(mask, noisy_spec)

            # Also return predicted wav
            predict_wav = self.hparams.resynth(predict_spec, noisy_wav)

            return predict_wav, mask

    def compute_objectives(self, predictions, batch, stage, optim_name=""):
        "Given the network predictions and targets compute the total loss"
        if self.sub_stage == SubStage.HISTORICAL:
            predict_wav = predictions
        else:
            predict_wav, mask = predictions

        predict_spec = self.compute_feats(predict_wav)

        ids = self.compute_ids(batch.id, optim_name)

        if self.hparams.use_dynamic_rt60:
            vis_loss = None
            no_vis_loss = None

        if self.sub_stage != SubStage.HISTORICAL:
            if self.use_soundspaces:
                clean_wav, _ = batch.clean_sig
            noisy_wav, lens = batch.noisy_sig
            src_rgb, _ = batch.rgb


        if optim_name == "generator":
            est_score = self.est_score(predict_spec)
            target_score = self.hparams.target_score * torch.ones(
                est_score.shape[0], 1, device=self.device
            )

            noisy_wav, lens = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)
            if self.use_soundspaces and self.hparams.supervised:
                clean_wav, lens = batch.clean_sig
                clean_spec = self.compute_feats(clean_wav)
                mse_cost = self.hparams.compute_cost(predict_spec, clean_spec, lens)
            else:
                mse_cost = self.hparams.compute_cost(predict_spec, noisy_spec, lens)

        # D Learns to estimate the scores of enhanced speech
        elif optim_name == "D_enh" and self.sub_stage == SubStage.CURRENT:
            target_score = self.score(
                ids, predict_wav, noisy_wav, lens, src_rgb
            )  # no clean_wav is needed
            est_score = self.est_score(predict_spec)

            if self.hparams.use_dynamic_rt60:
                revb_wav, lens = batch.revb_sig
                vis_loss, no_vis_loss = self.process_target_network(ids, predict_wav, revb_wav, lens, src_rgb)

            # Write enhanced wavs during discriminator training, because we
            # compute the actual score here and we can save it
            self.write_wavs(ids, predict_wav, target_score, lens)

        # D Relearns to estimate the scores of previous epochs
        elif optim_name == "D_enh" and self.sub_stage == SubStage.HISTORICAL:
            target_score = batch.score.unsqueeze(1).float()
            est_score = self.est_score(predict_spec)

        # D Learns to estimate the scores of noisy speech
        elif optim_name == "D_noisy":
            noisy_spec = self.compute_feats(noisy_wav)
            target_score = self.score(
                ids, noisy_wav, noisy_wav, lens, src_rgb
            )  # no clean_wav is needed
            est_score = self.est_score(noisy_spec)
            # Save scores of noisy wavs
            self.save_noisy_scores(ids, target_score)

        # D Learns to estimate the scores of clean speech
        elif optim_name == "D_clean":
            clean_spec = self.compute_feats(clean_wav)
            target_score = self.score(
                ids, clean_wav, clean_wav, lens, src_rgb
            )
            est_score = self.est_score(clean_spec)
            # Save scores of clean wavs
            self.save_noisy_scores(ids, target_score)

        if self.hparams.use_dynamic_rt60 and (vis_loss is not None) and (no_vis_loss is not None):
                self.metrics["vis"].append(vis_loss)
                self.metrics["no_vis"].append(no_vis_loss)

        if stage == sb.Stage.TRAIN:
            # Compute the cost
            cost = self.hparams.compute_cost(est_score, target_score)
            if optim_name == "generator":
                cost += self.hparams.mse_weight * mse_cost
                self.metrics["G"].append(cost.detach())
            else:
                self.metrics["D"].append(cost.detach())

        # Compute scores on validation data
        if stage != sb.Stage.TRAIN:
            noisy_wav, lens = batch.noisy_sig
            src_rgb, _ = batch.rgb

            cost = self.hparams.compute_si_snr(predict_wav, noisy_wav, lens) #FIX THIS!!
            #cost = srmrpy_eval(predict_wav, noisy_wav) #noisy_wav not actually used in srmr

            # Evaluate speech quality/intelligibility
            self.srmr_metric.append(
                batch.id,
                predict=predict_wav,
                target=predict_wav,
                lengths=lens,  # no clean_wav is needed
            )
            self.rt60_bias_metric.append(
                batch.id,
                predict=predict_wav,
                target=noisy_wav,
                rgb = src_rgb,
                lengths=lens,  # no clean_wav is needed
            )
            if (
                self.hparams.calculate_dnsmos_on_validation_set
            ):  # Note: very time consuming........
                self.dnsmos_metric.append(
                    batch.id,
                    predict=predict_wav,
                    target=predict_wav,
                    lengths=lens,  # no clean_wav is needed
                )

            # Write wavs to file, for evaluation
            lens = lens * noisy_wav.shape[1]
            for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                name += ".wav"
                enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                torchaudio.save(
                    enhance_path,
                    torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                    16000,
                )

        return cost

    def compute_ids(self, batch_id, optim_name):
        """Returns the list of ids, edited via optimizer name."""
        if optim_name == "D_enh":
            return [f"{uid}@{self.epoch}" for uid in batch_id]
        return batch_id

    def save_noisy_scores(self, batch_id, scores):
        for i, score in zip(batch_id, scores):
            self.noisy_scores[i] = score

    def process_target_network(self, batch_id, deg_wav, ref_wav, lens, src_rgb):
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]

        if (len(new_ids) != 0) and (self.hparams.target_metric == "rt60_bias"):
            predict=deg_wav[new_ids].detach()
            target=ref_wav[new_ids].detach()
            rgb=src_rgb[new_ids].detach()
            vis_loss, no_vis_loss = self.update_target_networks(predict, target, rgb)
        else:
            vis_loss = None
            no_vis_loss = None

        return vis_loss, no_vis_loss

    def score(self, batch_id, deg_wav, ref_wav, lens, src_rgb):
        """Returns actual metric score, either pesq or stoi
        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        deg_wav : torch.Tensor
            The degraded waveform to score
        ref_wav : torch.Tensor
            The reference waveform to use for scoring
        length : torch.Tensor
            The relative lengths of the utterances
        """
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]

        if len(new_ids) == 0:
            pass
        if self.hparams.target_metric == ("srmr" or "dnsmos" or "mosnet"):
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[
                    new_ids
                ].detach(),  # target is not used in the function !!!
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device,
            )
        elif self.hparams.target_metric == "rt60_bias":
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[
                    new_ids
                ].detach(),
                rgb=src_rgb[new_ids].detach(),  # target is not used in the function !!!
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device,
            )
        else:
            raise ValueError("Expected 'srmr' or 'dnsmos' or 'mosnet' or 'rt60_bias' for target_metric")

        # Clear metric scores to prepare for next batch
        self.target_metric.clear()

        # Combine old scores and new
        final_score = []
        for i, d in enumerate(batch_id):
            if d in self.historical_set:
                final_score.append([self.historical_set[d]["score"]])
            elif d in self.noisy_scores:
                final_score.append([self.noisy_scores[d]])
            else:
                final_score.append([score[new_ids.index(i)]])

        return torch.tensor(final_score, device=self.device)

    def est_score(self, deg_spec):
        """Returns score as estimated by discriminator

        Arguments
        ---------
        deg_spec : torch.Tensor
            The spectral features of the degraded utterance
        ref_spec : torch.Tensor
            The spectral features of the reference utterance
        """

        """
        combined_spec = torch.cat(
            [deg_spec.unsqueeze(1), ref_spec.unsqueeze(1)], 1
        )
        """
        return self.modules.discriminator(deg_spec.unsqueeze(1))

    def write_wavs(self, batch_id, wavs, score, lens):
        """Write wavs to files, for historical discriminator training

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        wavs : torch.Tensor
            The wavs to write to files
        score : torch.Tensor
            The actual scores for the corresponding utterances
        lens : torch.Tensor
            The relative lengths of each utterance
        """
        lens = lens * wavs.shape[1]
        record = {}
        for i, (name, pred_wav, length) in enumerate(zip(batch_id, wavs, lens)):
            path = os.path.join(self.hparams.MetricGAN_folder, name + ".wav")
            data = torch.unsqueeze(pred_wav[: int(length)].cpu(), 0)
            torchaudio.save(path, data, self.hparams.Sample_rate)

            # Make record of path and score for historical training

            new_score = float(score[i][0])
            record[name] = {
                "enh_wav": path,
                "score": new_score,
            }

        # Update records for historical training
        self.historical_set.update(record)

        with open(self.hparams.historical_file, "wb") as fp:  # Pickling
            pickle.dump(self.historical_set, fp)

    def fit_batch(self, batch):
        "Compute gradients and update either D or G based on sub-stage."

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)

        loss_tracker = 0
        if self.sub_stage == SubStage.CURRENT:
            if self.hparams.supervised:
                modes = ["enh", "noisy", "clean"]
            else:
                modes = ["enh", "noisy"]

            for mode in modes:
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN, f"D_{mode}"
                ).float()
                self.d_optimizer.zero_grad()
                loss.backward()
                if self.check_gradients(loss):
                    self.d_optimizer.step()
                loss_tracker += loss.detach() / 3
        elif self.sub_stage == SubStage.HISTORICAL:
            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "D_enh"
            )
            self.d_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.d_optimizer.step()
            loss_tracker += loss.detach()
        elif self.sub_stage == SubStage.GENERATOR:
            for name, param in self.modules.generator.named_parameters():
                if "Learnable_sigmoid" in name:
                    param.data = torch.clamp(
                        param, max=3.5
                    )  # to prevent gradient goes to infinity

            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "generator"
            )
            self.g_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.g_optimizer.step()
            loss_tracker += loss.detach()

        return loss_tracker

    def on_stage_start(self, stage, epoch=None):

        """Gets called at the beginning of each epoch

        This method calls ``fit()`` again to train the discriminator
        before proceeding with generator training.
        """

        self.metrics = {"G": [], "D": []}
        if self.hparams.use_dynamic_rt60:
            self.metrics = {"G": [], "D": [], "vis": [], "no_vis": []}

        if stage == sb.Stage.TRAIN:
            if self.hparams.target_metric == "srmr":
                self.target_metric = MetricStats(
                    metric=srmrpy_eval,
                    n_jobs=hparams["n_jobs"],
                    batch_eval=False,
                )
            elif self.hparams.target_metric == "dnsmos":
                self.target_metric = MetricStats(
                    metric=dnsmos_eval,
                    n_jobs=hparams["n_jobs"],
                    batch_eval=False,
                )
            elif self.hparams.target_metric == "mosnet":
                self.target_metric = MetricStats(
                    metric=mosnet_eval,
                    n_jobs=hparams["n_jobs"],
                    batch_eval=False,
                )
            elif self.hparams.target_metric == "rt60_bias":
                self.target_metric = MetricStats(
                    metric=self.rt60_bias_eval,
                    n_jobs=hparams["n_jobs"],
                    batch_eval=hparams["smoothed_metric"],
                )
            else:
                raise NotImplementedError(
                    "Right now we only support 'srmr', 'dnsmos', and 'rt60_bias"
                )

            # Train discriminator before we start generator training
            if self.sub_stage == SubStage.GENERATOR:
                self.epoch = epoch
                self.train_discriminator()
                self.sub_stage = SubStage.GENERATOR
                print("Generator training by current data...")

        if stage != sb.Stage.TRAIN:
            self.srmr_metric = MetricStats(
                metric=srmrpy_eval_valid,
                n_jobs=hparams["n_jobs"],
                batch_eval=False,
            )
            self.dnsmos_metric = MetricStats(
                metric=dnsmos_eval_valid,
                n_jobs=hparams["n_jobs"],
                batch_eval=False,
            )
            self.rt60_bias_metric = MetricStats(
                metric=self.rt60_bias_eval_valid,
                n_jobs=hparams["n_jobs"],
                batch_eval=False,
            )

    def train_discriminator(self):
        """A total of 3 data passes to update discriminator."""
        if self.hparams.use_dynamic_rt60:
            self.create_rt60_bias_copy_networks(lr=float(self.hparams.dynamic_rt60_lr), use_devb=self.hparams.use_devb, use_soundspaces=self.use_soundspaces)

        # First, iterate train subset w/ updates for enh, noisy
        print("Discriminator training by current data...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

        # Next, iterate historical subset w/ updates for enh
        if self.historical_set:
            print("Discriminator training by historical data...")
            self.sub_stage = SubStage.HISTORICAL
            self.fit(
                range(1),
                self.historical_set,
                train_loader_kwargs=self.hparams.dataloader_options,
            )


        #don't train target networks on current data again
        if self.hparams.use_dynamic_rt60:
            self.vis_reverb_tgt.eval()
            self.no_vis_reverb_tgt.eval()


        # Finally, iterate train set again. Should iterate same
        # samples as before, due to ReproducibleRandomSampler
        print("Discriminator training by current data again...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

    def on_stage_end(self, stage, stage_loss, epoch=None):

        if self.hparams.use_dynamic_rt60 and (len(self.metrics["vis"]) != 0) and (len(self.metrics["no_vis"]) != 0):
            self.vis_loss = torch.tensor(self.metrics["vis"]).float()
            self.no_vis_loss = torch.tensor(self.metrics["no_vis"]).float()
            print("Avg visual reverberator loss: %.3f" % torch.mean(self.vis_loss))
            print("Avg blind reverberator loss: %.3f" % torch.mean(self.no_vis_loss))
        else:
            self.vis_loss = None
            self.no_vis_loss = None

        "Called at the end of each stage to summarize progress"
        if self.sub_stage != SubStage.GENERATOR:
            return

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            g_loss = torch.tensor(self.metrics["G"])  # batch_size
            d_loss = torch.tensor(self.metrics["D"])  # batch_size
            print("Avg G loss: %.3f" % torch.mean(g_loss))
            print("Avg D loss: %.3f" % torch.mean(d_loss))
        else:
            if self.hparams.calculate_dnsmos_on_validation_set:
                stats = {
                    "SI-SNR": -stage_loss,
                    "srmr": self.srmr_metric.summarize("average"),
                    "dnsmos": self.dnsmos_metric.summarize("average"),
                    "rt60_bias": self.rt60_bias_metric.summarize("average"),
                }
            else:
                stats = {
                    "SI-SNR": -stage_loss,
                    "srmr": self.srmr_metric.summarize("average"),
                    "rt60_bias": self.rt60_bias_metric.summarize("average"),
                }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(5.0 - stats[self.hparams.target_metric])

            sb.nnet.schedulers.update_learning_rate(self.g_optimizer, new_lr)

            if self.hparams.use_tensorboard:
                if (
                    self.hparams.calculate_dnsmos_on_validation_set
                ):  # Note: very time consuming........
                    valid_stats = {
                        "SI-SNR": -stage_loss,
                        "srmr": self.srmr_metric.summarize("average"),
                        "dnsmos": self.dnsmos_metric.summarize("average"),
                        "rt60_bias": self.rt60_bias_metric.summarize("average"),
                    }
                else:
                    valid_stats = {
                        "SI-SNR": -stage_loss,
                        "srmr": self.srmr_metric.summarize("average"),
                        "rt60_bias": self.rt60_bias_metric.summarize("average"),
                    }

                self.hparams.tensorboard_train_logger.log_stats(
                    {"lr": old_lr}, valid_stats
                )

            if self.hparams.use_dynamic_rt60 and (self.vis_loss is not None) and (self.no_vis_loss is not None):
                self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss, "vis": torch.mean(self.vis_loss), "no vis": torch.mean(self.no_vis_loss)},
                valid_stats=stats,
                )
            else:
                self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
                )

            #self.checkpointer.save_and_keep_only(meta=stats, max_keys=["srmr"])
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=[self.hparams.target_metric])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        "Override dataloader to insert custom sampler/dataset"
        if stage == sb.Stage.TRAIN:

            # Create a new dataset each time, this set grows
            if self.sub_stage == SubStage.HISTORICAL:
                dataset = sb.dataio.dataset.DynamicItemDataset(
                    data=dataset,
                    dynamic_items=[enh_pipeline],
                    output_keys=["id", "enh_sig", "score"],
                )
                samples = round(len(dataset) * self.hparams.history_portion)
            else:
                samples = self.hparams.number_of_samples

            # This sampler should give the same samples for D and G
            epoch = self.hparams.epoch_counter.current

            # Equal weights for all samples, we use "Weighted" so we can do
            # both "replacement=False" and a set number of samples, reproducibly
            weights = torch.ones(len(dataset))
            sampler = ReproducibleWeightedRandomSampler(
                weights, epoch=epoch, replacement=False, num_samples=samples
            )
            loader_kwargs["sampler"] = sampler

            if self.sub_stage == SubStage.GENERATOR:
                self.train_sampler = sampler

        # Make the dataloader as normal
        return super().make_dataloader(
            dataset, stage, ckpt_prefix, **loader_kwargs
        )

    def on_fit_start(self):
        "Override to prevent this from running for D training"
        if self.sub_stage == SubStage.GENERATOR:
            super().on_fit_start()

    def init_optimizers(self):
        "Initializes the generator and discriminator optimizers"
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("g_opt", self.g_optimizer)
            self.checkpointer.add_recoverable("d_opt", self.d_optimizer)

    def transcribe_dataset(
        self,
        dataset, # Must be obtained from the dataio_function
        max_key, # We load the model with the highest SRMR
        loader_kwargs # opts for the dataloading
    ):

        # If dataset isn't a Dataloader, we create it.
        if not isinstance(dataset, torch.utils.data.DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        self.on_evaluate_start(max_key=max_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        if self.hparams.target_metric == "srmr" and not self.use_soundspaces:
            ds = "acoustic_avspeech"
            dvb_folder = "metric_debias" if self.hparams.use_devb else "metric_dereverb"
            loc_folder = os.path.join("data", ds, dvb_folder, "audio")
            print(loc_folder)
            existing_files = os.listdir(loc_folder) if os.path.isdir(loc_folder) else []
            existing_ids = [fil.split(".wav")[0] for fil in existing_files]


        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            ids = []
            preds = []
            i = 0
            print("Generating dataset...")
            for batch in tqdm.tqdm(dataset, dynamic_ncols=True):

                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                if self.hparams.target_metric == "srmr" and not self.use_soundspaces:
                    if all(item in existing_ids for item in batch.id):
                        continue

                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                pred_wav, mask = out

                ids.append(batch.id)
                preds.append(pred_wav.detach().cpu())

        return ids, preds

    def rt60_bias_eval(self, predict, target, rgb, lengths=None):
        """
            Normalize the score to 0~1 for training.
        """
        if self.hparams.smoothed_metric:
            if lengths is not None and len(lengths) == 0:
                return []

            try:
                batch = construct_batch(predict, target, rgb)
                vis_rt60, non_vis_rt60, tgt_rt60 = self.compute_rt60_err(batch, self.device)
            except IndexError:
                return [0]*len(lengths)

            eps = 1e-10
            if self.hparams.bias_variant == "diff_norm":
                tgt_rt60_clipped = max(tgt_rt60, 0.1)
                vis_err = (((abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)) / abs(tgt_rt60_clipped)).mean()).item()
                #vis_err = (abs(abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)).mean() / abs(tgt_rt60).mean()).item()
            elif self.hparams.bias_variant == "diff":
                vis_err = (abs(abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)).mean()).item()
            else:
                vis_err = (abs(non_vis_rt60-tgt_rt60).mean()/(abs(vis_rt60-tgt_rt60).mean()+eps)).item()

            if self.hparams.use_combined_loss:
                srmr_val = np.array([float(
                    srmr(
                    pred.detach().cpu().numpy(),
                    fs=16000,
                    n_cochlear_filters=23,
                    low_freq=125,
                    min_cf=4,
                    max_cf=128,
                    fast=True,
                    norm=False,
                    )[0]
                ) for pred in predict]).mean()

                return [float(sigmoid(0.1*vis_err + 0.1*srmr_val))] * len(lengths)
                #return [float(sigmoid(0.2*float(np.tanh(vis_err))  + 0.8*srmrpy_eval(predict, target)))] * len(lengths)
            else:
                return [float(sigmoid(0.1*vis_err))] * len(lengths)
        else:
            try:
                batch = construct_batch(predict, target, rgb)
                vis_rt60, non_vis_rt60, tgt_rt60 = self.compute_rt60_err(batch, self.device)
            except IndexError:
                return 0

            eps = 1e-10
            if self.hparams.bias_variant == "diff_norm":
                tgt_rt60_clipped = max(tgt_rt60, 0.1)
                vis_err = (abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)) / abs(tgt_rt60_clipped)
            elif self.hparams.bias_variant == "diff":
                vis_err = abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)
            elif self.hparams.bias_variant == "vis_rte":
                tgt_rt60_clipped = max(tgt_rt60, 0.1)
                vis_err = abs(vis_rt60-tgt_rt60) / abs(tgt_rt60_clipped)
            else:
                vis_err = abs(non_vis_rt60-tgt_rt60)/(abs(vis_rt60-tgt_rt60)+eps)

            if self.hparams.use_combined_loss:
                srmr_val = float(
                    srmr(
                    predict.numpy(),
                    fs=16000,
                    n_cochlear_filters=23,
                    low_freq=125,
                    min_cf=4,
                    max_cf=128,
                    fast=True,
                    norm=False,
                    )[0]
                )
                if self.hparams.bias_variant != "vis_rte":
                    return float(0.7*sigmoid(vis_err) + 0.3*sigmoid(0.1*srmr_val))
                else:
                    return float(0.7*(1-sigmoid(vis_err)) + 0.3*sigmoid(0.1*srmr_val))
            else:
                return float(sigmoid(vis_err)) if self.hparams.bias_variant != "vis_rte" else 1 - float(sigmoid(vis_err))

    def rt60_bias_eval_valid(self, predict, target, rgb):

        try:
            batch = construct_batch(predict, target, rgb)
            vis_rt60, non_vis_rt60, tgt_rt60 = self.compute_rt60_err(batch, self.device, eval_valid=True)
        except IndexError:
            return 0

        eps = 1e-10
        if self.hparams.bias_variant == "diff_norm":
            tgt_rt60_clipped = max(tgt_rt60, 0.1)
            vis_err = (abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)) / abs(tgt_rt60_clipped)
        elif self.hparams.bias_variant == "diff":
            vis_err = abs(non_vis_rt60-tgt_rt60) - abs(vis_rt60-tgt_rt60)
        elif self.hparams.bias_variant == "vis_rte":
            tgt_rt60_clipped = max(tgt_rt60, 0.1)
            vis_err = abs(vis_rt60-tgt_rt60) / abs(tgt_rt60_clipped)
        else:
            vis_err = abs(non_vis_rt60-tgt_rt60)/(abs(vis_rt60-tgt_rt60)+eps)
        return float(vis_err)

    def compute_rt60_err(self, features, device, eval_valid=False):
        with torch.no_grad():
            vis_pred = self.vis_reverb_copy.audio_synthesis(features, 0)['pred']
            no_vis_pred = self.no_vis_reverb_copy.audio_synthesis(features, 0)['pred']

            if self.hparams.smoothed_metric and not eval_valid:
                if vis_pred.shape[0] > 1:
                    vis_pred = vis_pred.squeeze()
                if no_vis_pred.shape[0] > 1:
                    no_vis_pred = no_vis_pred.squeeze()

            pred_rt60_vis = self.vis_reverb_copy.estimate_rt60(self.rt60_estimator, vis_pred.squeeze(0)).detach().cpu()
            pred_rt60_non_vis = self.no_vis_reverb_copy.estimate_rt60(self.rt60_estimator, no_vis_pred.squeeze(0)).detach().cpu()
            tgt_rt60 = self.no_vis_reverb_copy.estimate_rt60(self.rt60_estimator, features['tgt_wav']).detach().cpu()

            if eval_valid or (not self.hparams.smoothed_metric):
                pred_rt60_vis = pred_rt60_vis[0].item()
                pred_rt60_non_vis = pred_rt60_non_vis[0].item()
                tgt_rt60 = tgt_rt60[0].item()

            return pred_rt60_vis, pred_rt60_non_vis, tgt_rt60

    def load_vis_reverberator(self, device, lr=None, use_devb=False, use_soundspaces=False):
        from trainer import parser
        from novas.models.wavenet import WaveNet
        if use_devb:
            if use_soundspaces:
                if self.hparams.direct_opt:
                    pretrained_weights = 'data/models/novas_de_re_vis/v1/best_val.ckpt'
                else:
                    pretrained_weights = 'data/models/ss_srmr_vis/v1/best_val.ckpt'
            else:
                sspeech_pretrained = False
                if str(self.hparams.seed)[0] == "9":
                    pretrained_weights = 'data/models/av_stage1_ft_vis/v1/best_val.ckpt' #soundspaces pre-trained
                else:
                    pretrained_weights = 'data/models/av_stage_2_srmr_vis/v1/best_val.ckpt' #full pipeline avspeech
        else:
            if use_soundspaces:
                pretrained_weights = 'data/models/ss_re_vis_mgan/v1/best_val.ckpt'
            else:
                pretrained_weights = 'data/models/novas_av_mgan_vis/v1/initialize.ckpt'
        pretrained_weights = getattr(self.hparams, "visual_reverberator_checkpoint", None) or pretrained_weights
        state_dict = torch.load(pretrained_weights, map_location='cpu')

        default_args = parser.parse_args("")
        update_args(default_args,
                vars(state_dict['hparams']), lr)
        reverberator = WaveNet(default_args).to(device=device)
        reverberator.load_state_dict(state_dict['state_dict'])
        optimizer = reverberator.configure_optimizers()['optimizer']

        reverberator.eval()
        return reverberator, optimizer

    def load_no_vis_reverberator(self, device, lr=None, use_devb=False, use_soundspaces=False):
        from trainer import parser
        from novas.models.wavenet import WaveNet
        if use_devb:
            if use_soundspaces:
                if self.hparams.direct_opt:
                    pretrained_weights = 'data/models/novas_de_re_no_vis/v1/best_val.ckpt'
                else:
                    pretrained_weights = 'data/models/ss_srmr_no_vis/v1/best_val.ckpt'
            else:
                sspeech_pretrained = False
                if str(self.hparams.seed)[0] == "9":
                    pretrained_weights = 'data/models/av_stage1_ft_no_vis/v1/best_val.ckpt' #soundspaces pre-trained
                else:
                    pretrained_weights = 'data/models/av_stage_2_srmr_no_vis/v1/best_val.ckpt' #full pipeline avspeech
        else:
            if use_soundspaces:
                pretrained_weights = 'data/models/ss_re_no_vis_mgan/v1/best_val.ckpt'
            else:
                pretrained_weights = 'data/models/novas_av_mgan_no_vis/v1/initialize.ckpt'
        pretrained_weights = getattr(self.hparams, "blind_reverberator_checkpoint", None) or pretrained_weights
        state_dict = torch.load(pretrained_weights, map_location='cpu')

        default_args = parser.parse_args("")
        update_args(default_args,
                vars(state_dict['hparams']), lr)
        reverberator = WaveNet(default_args).to(device=device)
        reverberator.load_state_dict(state_dict['state_dict'])
        optimizer = reverberator.configure_optimizers()['optimizer']

        reverberator.eval()
        return reverberator, optimizer

    def update_target_networks(self, pred, tgt, rgb):

        batch = construct_batch(pred, tgt, rgb)

        loss_vis = self.vis_reverb_tgt.training_step(batch, 0)
        self.vis_reverb_opt.zero_grad()
        loss_vis['loss'].backward()
        self.vis_reverb_opt.step()

        loss_no_vis = self.no_vis_reverb_tgt.training_step(batch, 0)
        self.no_vis_reverb_opt.zero_grad()
        loss_no_vis['loss'].backward()
        self.no_vis_reverb_opt.step()

        return loss_vis['loss'].item(), loss_no_vis['loss'].item()

    def create_rt60_bias_copy_networks(self, lr=None, use_devb=False, use_soundspaces=False):

        if (self.epoch == 1) or (self.epoch % self.hparams.update_freq == 0):
            self.vis_reverb_copy = copy.deepcopy(self.vis_reverb_tgt)
            self.vis_reverb_copy.to(self.device) #Don't need this??
            self.vis_reverb_copy.eval()

            self.no_vis_reverb_copy = copy.deepcopy(self.no_vis_reverb_tgt)
            self.no_vis_reverb_copy.to(self.device) #Don't need this??
            self.no_vis_reverb_copy.eval()

            self.save_vis_model(self.epoch, self.hparams.seed, use_devb=self.hparams.use_devb, use_soundspaces=self.use_soundspaces)
        self.vis_reverb_tgt.train()
        self.no_vis_reverb_tgt.train()

    def save_vis_model(self, epoch, seed, use_devb=False, use_soundspaces=False):
        if use_devb:
            if use_soundspaces:
                if self.hparams.direct_opt:
                    model_dir = os.path.join('data/models/novas_de_re_vis/rt60_finetuned/', str(seed))
                    non_vis_model_dir = os.path.join('data/models/novas_de_re_no_vis/rt60_finetuned/', str(seed))
                else:
                    model_dir = os.path.join('data/models/ss_srmr_vis/rt60_finetuned/', str(seed))
                    non_vis_model_dir = os.path.join('data/models/ss_srmr_no_vis/rt60_finetuned/', str(seed))
            else:
                sspeech_pretrained = False
                if str(self.hparams.seed)[0] == "9":
                    model_dir = os.path.join('data/models/av_stage1_ft_vis/rt60_finetuned/', str(seed)) #soundspaces pre-trained
                    non_vis_model_dir = os.path.join('data/models/av_stage1_ft_no_vis/rt60_finetuned/', str(seed))
                else:
                    model_dir = os.path.join('data/models/av_stage_2_srmr_vis/rt60_finetuned/', str(seed)) #full pipeline avspeech
                    non_vis_model_dir = os.path.join('data/models/av_stage_2_srmr_no_vis/rt60_finetuned/', str(seed))
        else:
            if use_soundspaces:
                model_dir = os.path.join('data/models/ss_re_vis_mgan/rt60_finetuned/', str(seed))
            else:
                model_dir = os.path.join('data/models/novas_av_mgan_vis/rt60_finetuned/', str(seed))
        os.makedirs(model_dir, exist_ok=True)
        epoch_save = "checkpoint_" + str(epoch) + ".ckpt"
        ckpt_path = os.path.join(model_dir, epoch_save)
        torch.save({'state_dict': self.vis_reverb_copy.state_dict(), 'hparams': self.vis_reverb_copy.args}, ckpt_path)
        print("Saving model in", ckpt_path)

        os.makedirs(non_vis_model_dir, exist_ok=True)
        epoch_save = "best_val.ckpt"
        ckpt_path = os.path.join(non_vis_model_dir, epoch_save)
        torch.save({'state_dict': self.no_vis_reverb_copy.state_dict(), 'hparams': self.no_vis_reverb_copy.args}, ckpt_path)
        print("Saving model in", ckpt_path)

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError

def read_rgb(rgb_obj):
    rgb_obj = rgb_obj.replace(".wav", ".png")
    rgb = np.array(Image.open(rgb_obj))
    rgb_res = (192,576)
    if rgb.shape[:2] != rgb_res:
        rgb = torchvision.transforms.Resize(rgb_res)(to_tensor(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    rgb = rgb.astype(np.float32)

    rgb_proc = to_tensor(rgb).permute(2, 0, 1) / 255.0
    return rgb_proc

def read_sspeech_audio(fp, mode="noisy"):
    with open(fp, 'rb') as fo:
        data = pickle.load(fo)
    if mode == "noisy":
        audio = data['receiver_audio'][:40960]
    else:
        audio = data['source_audio'][:40960]

    audio = normalize(audio, norm='peak')
    if audio.shape[0] < 40960:
        audio = np.pad(audio, (0, 40960 - audio.shape[0]))
    return to_tensor(audio)

def read_sspeech_rgb(fp):
    with open(fp, 'rb') as fo:
        data = pickle.load(fo)
    rgb = to_tensor(np.concatenate([x / 255.0 for x in data['rgb']], axis=1)).permute(2, 0, 1)
    depth = to_tensor(np.concatenate(data['depth'], axis=1)).unsqueeze(0)
    rgb = torchvision.transforms.Resize((192, 576))(rgb)
    depth = torchvision.transforms.Resize((192, 576))(depth)

    visual = torch.cat((rgb, depth), dim=0)
    return visual

# Define audio piplines for training set on devb data
@sb.utils.data_pipeline.takes("devb_wav", "src_rgb", "noisy_wav")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "revb_sig")
def audio_pipeline_devb_train(devb_wav, rgb, revb_wav):
    yield sb.dataio.dataio.read_audio(devb_wav)
    yield read_rgb(rgb)
    yield sb.dataio.dataio.read_audio(revb_wav)

# Define audio piplines for validation/test set on devb data
@sb.utils.data_pipeline.takes("devb_wav", "src_rgb", "noisy_wav")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "revb_sig")
def audio_pipeline_devb_valid(devb_wav, rgb, revb_wav):
    yield sb.dataio.dataio.read_audio(devb_wav)
    yield read_rgb(rgb)
    yield sb.dataio.dataio.read_audio(revb_wav)

# Define audio piplines for training set
@sb.utils.data_pipeline.takes("noisy_wav", "src_rgb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb")
def audio_pipeline_train(noisy_wav, rgb):
    yield sb.dataio.dataio.read_audio(noisy_wav)
    yield read_rgb(rgb)

# Define audio piplines for validation/test set
@sb.utils.data_pipeline.takes("noisy_wav", "src_rgb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb")
def audio_pipeline_valid(noisy_wav, rgb):
    yield sb.dataio.dataio.read_audio(noisy_wav)
    yield read_rgb(rgb)

# Define audio piplines for SoundSpaces training set
@sb.utils.data_pipeline.takes("data", "dereverb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "clean_sig")
def audio_pipeline_sspeech_train(ss_data, devb_wav):
    yield read_sspeech_audio(ss_data, "noisy")
    yield read_sspeech_rgb(ss_data)
    yield read_sspeech_audio(ss_data, "clean")

# Define audio piplines for SoundSpaces validation/test set
@sb.utils.data_pipeline.takes("data", "dereverb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "clean_sig")
def audio_pipeline_sspeech_valid(ss_data, devb_wav):
    yield read_sspeech_audio(ss_data, "noisy")
    yield read_sspeech_rgb(ss_data)
    yield read_sspeech_audio(ss_data, "clean")

# Define audio piplines for SoundSpaces devb training set
@sb.utils.data_pipeline.takes("data", "dereverb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "clean_sig", "revb_sig")
def audio_pipeline_devb_sspeech_train(ss_data, devb_wav):
    yield sb.dataio.dataio.read_audio(devb_wav)
    yield read_sspeech_rgb(ss_data)
    yield read_sspeech_audio(ss_data, "clean")
    yield read_sspeech_audio(ss_data, "noisy")

# Define audio piplines for SoundSpaces devb validation/test set
@sb.utils.data_pipeline.takes("data", "dereverb")
@sb.utils.data_pipeline.provides("noisy_sig", "rgb", "clean_sig", "revb_sig")
def audio_pipeline_devb_sspeech_valid(ss_data, devb_wav):
    yield sb.dataio.dataio.read_audio(devb_wav)
    yield read_sspeech_rgb(ss_data)
    yield read_sspeech_audio(ss_data, "clean")
    yield read_sspeech_audio(ss_data, "noisy")

# Define audio piplines for LibriSpeech generation
@sb.utils.data_pipeline.takes("data")
@sb.utils.data_pipeline.provides("noisy_sig")
def ls_pipeline(ss_data):
    yield read_sspeech_audio(ss_data, "clean")

# For historical data
@sb.utils.data_pipeline.takes("enh_wav")
@sb.utils.data_pipeline.provides("enh_sig")
def enh_pipeline(enh_wav):
    yield sb.dataio.dataio.read_audio(enh_wav)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class."""

    # Define datasets
    datasets = {}
    if not hparams["use_devb"]:
        if hparams["data_folder"] == "data/soundspaces_speech":
            datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["train_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_sspeech_train],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig"],
            )
            datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["valid_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig"],
            )
            datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["test_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig"],
            )
            datasets["full"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path="data/soundspaces_speech/full_revb.json",
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig"],
            )
        else:
            datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["train_annotation_sub"] if hparams['use_avspeech_subset'] else hparams["train_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_train],
                output_keys=["id", "noisy_sig", "rgb"],
            )
            datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["valid_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_valid],
                output_keys=["id", "noisy_sig", "rgb"],
            )
            datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["test_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_valid],
                output_keys=["id", "noisy_sig", "rgb"],
            )
            datasets["full"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path="data/acoustic_avspeech/full_sub_revb.json" if hparams['use_avspeech_subset'] else "data/acoustic_avspeech/full_revb.json",
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_valid],
                output_keys=["id", "noisy_sig", "rgb"],
            )
    else:
        if hparams["data_folder"] == "data/soundspaces_speech":
            datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["train_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_sspeech_train],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig", "revb_sig"],
            )
            datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["valid_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig", "revb_sig"],
            )
            datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["test_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig", "revb_sig"],
            )
            datasets["full"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path="data/soundspaces_speech/full_revb.json",
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_sspeech_valid],
                output_keys=["id", "noisy_sig", "rgb", "clean_sig", "revb_sig"],
            )
        else:
            datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["train_annotation_sub"] if hparams['use_avspeech_subset'] else hparams["train_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_train],
                output_keys=["id", "noisy_sig", "rgb", "revb_sig"],
            )
            datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["valid_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_valid],
                output_keys=["id", "noisy_sig", "rgb", "revb_sig"],
            )
            datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["test_annotation"],
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_valid],
                output_keys=["id", "noisy_sig", "rgb", "revb_sig"],
            )
            datasets["full"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path="data/acoustic_avspeech/full_sub_revb.json" if hparams['use_avspeech_subset'] else "data/acoustic_avspeech/full_revb.json",
                replacements={"data_root": hparams["data_folder"]},
                dynamic_items=[audio_pipeline_devb_valid],
                output_keys=["id", "noisy_sig", "rgb", "revb_sig"],
            )

    if os.path.isfile("data/soundspaces_speech/ls_revb.json"):
        datasets["librispeech"] = sb.dataio.dataset.DynamicItemDataset.from_json(
                    json_path="data/soundspaces_speech/ls_revb.json",
                    replacements={"data_root": "data/soundspaces_speech"},
                    dynamic_items=[ls_pipeline],
                    output_keys=["id", "noisy_sig"],
                    )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)


def save_dataset(ids, preds, hparams):
    return

# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    import argparse
    recipe_parser = argparse.ArgumentParser(add_help=False)
    recipe_parser.add_argument("--generate_dataset", action="store_true")
    recipe_parser.add_argument("--generate_libri_dataset", action="store_true")
    recipe_args, remaining = recipe_parser.parse_known_args()
    hparams_file, run_opts, overrides = sb.parse_arguments(remaining)
    run_opts.update(vars(recipe_args))

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    #identify dataset type
    ds_type = "vctk"

    if "vctk" in hparams["data_folder"]:
        ds_type = "vctk"
    if "avspeech" in hparams["data_folder"]:
        ds_type = "avspeech"
    if "soundspaces" in hparams["data_folder"]:
        ds_type = "soundspaces"

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    if ds_type == "vctk":
        from voicebank_revb_prepare import prepare_data
    elif ds_type == "avspeech":
        from avspeech_prepare import prepare_data
    elif ds_type == "soundspaces":
        from sspeech_prepare import prepare_data
    else:
        raise ValueError('Invalid Dataset')

    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})
    se_brain = MetricGanBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    se_brain.train_set = datasets["train"]
    se_brain.historical_set = {}
    se_brain.noisy_scores = {}
    se_brain.batch_size = hparams["dataloader_options"]["batch_size"]
    se_brain.sub_stage = SubStage.GENERATOR

    if not os.path.isfile(hparams["historical_file"]):
        try:
            shutil.rmtree(hparams["MetricGAN_folder"])
        except FileNotFoundError:
            pass
    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_folder"]})

    se_brain.load_history(run_opts["generate_dataset"])

    if run_opts["generate_dataset"] != True:
        # Load latest checkpoint to resume training
        se_brain.fit(
            epoch_counter=se_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["valid_dataloader_options"],
        )

        # Load best checkpoint for evaluation
        test_stats = se_brain.evaluate(
            test_set=datasets["test"],
            max_key=hparams["target_metric"],
            test_loader_kwargs=hparams["dataloader_options"],
        )
    else:
        # Create dereverberated dataset from best generator
        if run_opts["generate_libri_dataset"]:
            ids, preds = se_brain.transcribe_dataset(
                    dataset=datasets["librispeech"],
                    max_key=hparams["target_metric"],
                    loader_kwargs=hparams["dataloader_options"],
            )
        else:
            if hparams["target_metric"] == "srmr":
                ids, preds = se_brain.transcribe_dataset(
                    dataset=datasets["full"],
                    max_key=hparams["target_metric"],
                    loader_kwargs=hparams["dataloader_options"],
                )
            else:
                ids, preds = se_brain.transcribe_dataset(
                    dataset=datasets["full"] if hparams["generate_avspeech_dbs"] or hparams.get("export_all", False) else datasets["test"],
                    max_key=hparams["target_metric"],
                    loader_kwargs=hparams["dataloader_options"],
                )

        ids = [j for i in ids for j in i]
        preds = torch.cat(preds, 0).unsqueeze(1)

        if run_opts['generate_libri_dataset']:
            save_folder = "ls_debias"
        else:
            if hparams["use_devb"]:
                save_folder = "metric_debias"
            else:
                save_folder = "metric_dereverb"

        if hparams["use_dynamic_rt60"]:
            save_folder = save_folder + "/dynamic_rt60"
        elif hparams["target_metric"] == "rt60_bias":
            save_folder = save_folder + "/rt60"

        if hparams["generate_avspeech_dbs"]:
            assert hparams['data_folder'] == "data/acoustic_avspeech"
            assert hparams['use_devb']
            assert run_opts["generate_dataset"]
            print("Generating debiased avspeech...")
            save_data_folder = os.path.join(hparams['data_folder'], "avspeech_dbs", "audio")
        elif run_opts['generate_libri_dataset']:
            save_data_folder = os.path.join("data/soundspaces_speech", save_folder, "audio")
        else:
            save_data_folder = os.path.join(hparams['data_folder'], save_folder, "audio")

        print("saving generated speech to...:")
        print(save_data_folder)

        run_on_main(create_folder, kwargs={"folder": save_data_folder})

        for idx, id in enumerate(ids):
            audio_path = save_data_folder + "/" + id + ".wav"
            run_on_main(torchaudio.save, args=[audio_path, preds[idx], hparams["Sample_rate"]])
