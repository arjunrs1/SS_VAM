# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torch.nn as nn
#import torchaudio.transforms as TT

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from argparse import ArgumentParser
from dataset import from_path, from_gtzan
from model import DiffWave
from params import AttrDict

DEREVERBERATOR = None


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

def update_args(args, dic):
  for key, value in dic.items():
      setattr(args, key, value)
  return args

def load_dereverberator(device):
  from trainer import parser
  parser = ArgumentParser()
  from novas.models.wavenet import WaveNet
  pretrained_weights = 'data/pretrained-models/dereverb/best_val.ckpt'
  state_dict = torch.load(pretrained_weights, map_location='cpu')
  default_args = parser.parse_args("")
  update_args(default_args, 
          vars(state_dict['hparams']))
  dereverberator = WaveNet(default_args).to(device=device)
  dereverberator.load_state_dict(state_dict['state_dict'])
  dereverberator.eval()
  return dereverberator



class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, valid_dataset, optimizer,
   params, device, use_frozen_dereverb=False, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.valid_dataset = valid_dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.best_val_loss = np.inf
    self.use_frozen_dereverb = use_frozen_dereverb
    #self.device = device
    self.sr = 16000

    mel_args = {
            'sample_rate': self.sr,
            'win_length': params.hop_samples * 4,
            'hop_length': params.hop_samples,
            'n_fft': params.n_fft,
            'f_min': 20.0,
            'f_max': self.sr / 2.0,
            'n_mels': params.n_mels,
            'power': 1.0,
            'normalized': True,
        }
    #self.mel_spec_transform = TT.MelSpectrogram(**mel_args).to(device)


    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, from_pretrained=False):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    if from_pretrained:
      self.step = 0
    else:
      self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights', save_best=False):
    save_basename = f'{filename}-{self.step // len(self.dataset)}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    if save_best:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_pretrained(self, filename='weights'):
    try:
      checkpoint = torch.load(f'data/models/diff_test2/{filename}.pt')
      self.load_state_dict(checkpoint, from_pretrained=True)
      return True
    except FileNotFoundError:
      return False
  
  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    train_loss = 0.0
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        train_loss += loss.item()
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % len(self.dataset) == 0 and self.step >= 1:
            save_best = False
            val_loss = self.val_step()
            if val_loss < self.best_val_loss:
              self.best_val_loss = val_loss
              save_best = True
            self.save_to_checkpoint(save_best=save_best)
            self._write_summary(self.step // len(self.dataset), features, train_loss, val_loss)
            train_loss = 0.0
        self.step += 1
  
  def val_step(self):
    device = next(self.model.parameters()).device
    loss = 0.0
    for features in tqdm(self.valid_dataset, desc=f'Validation Epoch {self.step // len(self.dataset)}') if self.is_master else self.valid_dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        #with torch.no_grad??
        loss += self.train_step(features, validation=True).item()
    
    self.model.train()
    return loss


  def train_step(self, features, validation=False):
    for param in self.model.parameters():
      param.grad = None

    if validation:
      self.model.eval()
    

    if self.use_frozen_dereverb:
            dereverb_sspeech(features, self.device)

    #self.compute_spectrogram(features)

    input_audio = features['src_wav']
    tgt_audio = features['tgt_wav']
    spectrogram = features['spectrogram']

    N, T = input_audio.shape
    device = input_audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=input_audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(input_audio)
      noisy_audio = noise_scale_sqrt * input_audio + (1.0 - noise_scale)**0.5 * noise

      predicted = self.model(noisy_audio, t, spectrogram)
      loss = self.loss_fn(noise + (input_audio - tgt_audio), predicted.squeeze(1))  #modified loss function!

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, epoch, features, loss, val_loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=epoch)
    writer.add_audio('feature/input_audio', features['src_wav'][0], epoch, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), epoch)
    writer.add_scalar('train/loss', loss, epoch)
    writer.add_scalar('train/grad_norm', self.grad_norm, epoch)

    writer.add_scalar('val/loss', val_loss, epoch)

    writer.flush()
    self.summary_writer = writer

  def compute_spectrogram(self, features):
    with torch.no_grad():
      audio = features['src_wav']
      audio = torch.clamp(audio, -1.0, 1.0)
      spectrogram = self.mel_spec_transform(audio)
      spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
      spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
      
      start = random.randint(0, spectrogram.shape[2] - self.params.crop_mel_frames)
      end = start + self.params.crop_mel_frames
      spectrogram = spectrogram[:,:,start:end]
        
      features['spectrogram'] = spectrogram
      


def _train_impl(replica_id, model, dataset, valid_dataset, args, params, device=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, valid_dataset,
   opt, params, use_frozen_dereverb=args.use_frozen_dereverb, fp16=args.fp16, device=device)
  learner.is_master = (replica_id == 0)
  if args.use_pretrained:
    learner.restore_from_pretrained()
  else:
    learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  if args.dataset == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args, params, split=args.split)
    valid_dataset = from_path(args, params, split="val")
  
  model = DiffWave(params).cuda()
  _train_impl(0, model, dataset, valid_dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.dataset == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args, params, split=args.split, is_distributed=True)
    valid_dataset = from_path(args, params, split="val", is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, valid_dataset, args, params, device)

def dereverb_sspeech(features, device, randomize_input=False, use_vida=False, encode_wav=True):
  with torch.no_grad():
      global DEREVERBERATOR
      if DEREVERBERATOR is None:
          DEREVERBERATOR = load_dereverberator(device)
      
      if randomize_input:
          original_rgb = features['src_rgb']
          original_depth = features['src_depth']
          features['src_rgb'] = torch.roll(features['src_rgb'], shifts=1, dims=0)
          features['src_depth'] = torch.roll(features['src_depth'], shifts=1, dims=0)
      if use_vida:
          pred_clean = DEREVERBERATOR.dereverb(features, 0)['pred']
      else:
          pred_clean = DEREVERBERATOR.audio_synthesis(features, 0)['pred']
      if randomize_input:
          #restore original rgb tensor for use in reverberation model later
          features['src_rgb'] = original_rgb
          features['src_depth'] = original_depth
      if encode_wav:
          features['src_wav'] = torch.squeeze(pred_clean[:, :features['tgt_wav'].shape[1]])
      else:
          features['src_wav'] = pred_clean
