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

from pyroomacoustics.experimental.rt60 import measure_rt60


from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from argparse import ArgumentParser
from dataset import from_path, from_gtzan
from model import DiffWave
from params import AttrDict

DEREVERBERATOR = None
RT60_ESTIMATOR = None


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

def load_rt60_estimator(device):
    from novas.models.visual_net import VisualNet
    rt60_estimator = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
    pretrained_weights = 'data/pretrained-models/rt60_estimator.pth'
    rt60_estimator.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['predictor'])
    rt60_estimator.to(device=device).eval()
    return rt60_estimator

def load_dereverberator(device):
  from trainer import parser
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
  def __init__(self, model_dir, model, dataset, valid_dataset, test_dataset, optimizer,
   params, device, use_frozen_dereverb=False, use_rt60_feature=False, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.best_val_loss = np.inf
    self.use_frozen_dereverb = use_frozen_dereverb
    self.use_rt60_feature = use_rt60_feature
    self.device = device
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
    import torchaudio.transforms as TT
    self.mel_spec_transform = TT.MelSpectrogram(**mel_args).to(device)

    self.RT60_ESTIMATOR = load_rt60_estimator(self.device)

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
      self.model.module.load_state_dict(state_dict['model'], strict=False)
    else:
      self.model.load_state_dict(state_dict['model'], strict=True)
    #self.optimizer.load_state_dict(state_dict['optimizer'])
    #self.scaler.load_state_dict(state_dict['scaler'])
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
  
  def restore_best_val(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def restore_from_checkpoint(self, filename='weights'):
    try:
      file_list = os.listdir(self.model_dir)
      file_list = [file for file in file_list if file.endswith(".pt")]
      last_save = file_list[-1]
      checkpoint = torch.load(f'{self.model_dir}/{last_save}')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def estimate_rt60(self, estimator, wav):
    stft = torch.stft(wav, n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=self.device),
                      pad_mode='constant', return_complex=True)
    spec = torch.log1p(stft.abs()).unsqueeze(1)
    with torch.no_grad():
      estimated_rt60 = estimator(spec)
    return estimated_rt60

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
        loss += self.train_step(features, validation=True).item()
    
    self.model.train()
    return loss


  def train_step(self, features, validation=False):
    for param in self.model.parameters():
      param.grad = None

    if validation:
      self.model.eval()

    if self.use_rt60_feature:
      src_rt60 = self.estimate_rt60(self.RT60_ESTIMATOR, features['src_wav'])
    else:
      src_rt60 = None

    if self.use_frozen_dereverb:
            dereverb_sspeech(features, self.device)

    self.compute_spectrogram(features, src_rt60)

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

      predicted = self.model(noisy_audio, t, spectrogram, src_rt60)
      loss = self.loss_fn(noise + (input_audio - tgt_audio), predicted.squeeze(1))  #modified loss function!

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def test(self):
    device = next(self.model.parameters()).device
    self.model.eval()

    L1_loss = []
    RT60_error = []
    for features in tqdm(self.test_dataset, desc=f'Testing Epoch') if self.is_master else self.test_dataset:
      features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
      wav_loss, rt60_error, _ = self.test_step(features)
      #L1_loss += wav_loss.item()
      L1_loss.append(wav_loss.item())
      RT60_error.append(rt60_error.item())
      #RT60_error += rt60_error.item()
      if torch.isnan(wav_loss).any():
        raise RuntimeError(f'Detected NaN loss.')

    L1_loss = np.array(L1_loss)
    RT60_error = np.array(RT60_error)
    print("L1_loss:", str(float(L1_loss.mean())))
    print("RT60 Error:", str(float(RT60_error.mean())))
    return

  def test_step(self, features):
    from inference import predict

    tgt_rt60 = self.estimate_rt60(self.RT60_ESTIMATOR, features['tgt_wav'].to(self.device))

    if self.use_rt60_feature:
      src_rt60 = self.estimate_rt60(self.RT60_ESTIMATOR, features['src_wav'].to(self.device))
    else:
      src_rt60 = None

    if self.use_frozen_dereverb:
            dereverb_sspeech(features, self.device)

    self.compute_spectrogram(features, src_rt60)

    input_audio = features['src_wav']
    tgt_audio = features['tgt_wav']
    spectrogram = features['spectrogram']

    pred, sr = predict(spectrogram, src_rt60, model_dir=self.model_dir, params=self.params, device=self.device)
    pred_rt60 = self.estimate_rt60(self.RT60_ESTIMATOR, pred)

    L1_loss = self.loss_fn(input_audio, pred)
    RT60_error = (pred_rt60 - tgt_rt60).abs().mean().cpu().numpy()

    return L1_loss, RT60_error, pred



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

  def compute_spectrogram(self, features, rt60_feature=None):
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
      


def _train_impl(replica_id, model, dataset, valid_dataset, test_dataset, args, params, device=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, valid_dataset, test_dataset, 
  opt, params, use_frozen_dereverb=args.use_frozen_dereverb, use_rt60_feature=args.use_rt60_feature,
  fp16=args.fp16, device=device)
  learner.is_master = (replica_id == 0)
  if not args.test and not args.visualize:
    if args.use_pretrained and not os.path.exists(os.path.join(args.model_dir, "weights.pt")):
      learner.restore_from_pretrained()
    else:
      learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)
  else:
    learner.restore_best_val()
    if args.test:
      learner.test()
    if args.visualize:
      generate_qual(args, learner, test_dataset)


def train(args, params):
  if args.dataset == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args, params, split=args.split)
    valid_dataset = from_path(args, params, split="val")
    test_dataset = from_path(args, params, split=args.test_split)

  device = torch.device('cuda:0')
  torch.cuda.set_device(device)
  #model = DiffWave(params, args.use_rt60_feature).cuda()
  model = DiffWave(params, args.use_rt60_feature).to(device)
  _train_impl(0, model, dataset, valid_dataset, test_dataset, args, params, device=device)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.dataset == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args, params, split=args.split, is_distributed=True)
    valid_dataset = from_path(args, params, split="val", is_distributed=True)
    test_dataset = from_path(args, params, split=args.test_split, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params, args.use_rt60_feature).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
  _train_impl(replica_id, model, dataset, valid_dataset, test_dataset, args, params, device)

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

def measure_rt60_wrapper(x, sr=16000, decay_db=30):
    rt60 = -1
    while rt60 == -1:
        try:
            rt60 = measure_rt60(x, sr, decay_db)
        except ValueError:
            if decay_db > 10:
                decay_db -= 10
            else:
                rt60 = x.shape[0]

    return rt60

def generate_qual(args, model, dataset):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from librosa.display import waveshow
    from as_scripts.webify import webify

    qual_dir = os.path.join(args.model_dir, "qual")
    os.makedirs(qual_dir, exist_ok=True)
    count = 0
    
    for i, batch in enumerate(dataset):
        device = next(model.model.parameters()).device
        _, _, pred = model.test_step(batch)

        wav_dict = {
            'pred': pred,
            'tgt': batch['tgt_wav'],
            'input': batch['src_wav']
        }
        spec_dict = {}
        for key in wav_dict:
            wav = wav_dict[key].reshape(-1, wav_dict[key].shape[-1])
            spec = torch.log1p(torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                               window=torch.hamming_window(400, device=wav_dict[key].device), pad_mode='constant',
                               center=True, return_complex=True).abs())
            spec = spec[:, :spec.shape[1]//2, :]
            spec_dict[key] = spec.reshape(pred.shape[0], -1, *spec.shape[1:])

        for dic in [wav_dict, spec_dict]:
            for key, val in dic.items():
                dic[key] = val.cpu().numpy()

        for j in tqdm(range(pred.shape[0])):
            y_limit = max([np.abs(wav[j]).max() for wav in wav_dict.values()]) + 0.03
            audio_duration = wav_dict['input'].shape[-1] / args.sample_rate
            #step = 0.2 if args.audio_len == 16000 else 0.5
            step = 0.5
            x_ticks = np.arange(step, audio_duration, step=step)

            for k in range(1):
                fig, axes = plt.subplots(2, 3)
                axes[0][0].imshow(np.squeeze(spec_dict['input'][j]))
                axes[0][0].set_aspect(1 / axes[0][0].get_data_ratio())
                axes[0][0].invert_yaxis()
                axes[0][0].set_title('Input')

                plt.sca(axes[1][0])
                waveshow(wav_dict['input'][j], args.sample_rate)
                axes[1][0].set_ylim(-y_limit, y_limit)
                axes[1][0].set_xticks(x_ticks)

                axes[0][1].imshow(np.squeeze(spec_dict['pred'][j]))
                axes[0][1].set_aspect(1 / axes[0][1].get_data_ratio())
                axes[0][1].invert_yaxis()
                axes[0][1].set_title('Prediction')

                plt.sca(axes[1][1])
                waveshow(wav_dict['pred'][j], args.sample_rate)
                axes[1][1].set_ylim(-y_limit, y_limit)
                axes[1][1].set_xticks(x_ticks)

                axes[0][2].imshow(np.squeeze(spec_dict['tgt'][j]))
                axes[0][2].set_aspect(1 / axes[0][2].get_data_ratio())
                axes[0][2].invert_yaxis()
                axes[0][2].set_title('Target')

                plt.sca(axes[1][2])
                waveshow(wav_dict['tgt'][j], args.sample_rate)
                axes[1][2].set_ylim(-y_limit, y_limit)
                axes[1][2].set_xticks(x_ticks)

                fig.tight_layout()
                plt.savefig(os.path.join(qual_dir, f"{count}-plot.png"))
                plt.close(fig)

            if 'src_rgb' in batch:
                rgb = batch['src_rgb']
                rgb = rgb.cpu()[j].permute(1, 2, 0).clip(0, 1).numpy()  # L/R channels are displayed separately
                plt.imsave(os.path.join(qual_dir, f'{count}-src-rgb.png'), rgb)
            if 'tgt_rgb' in batch:
                rgb = batch['tgt_rgb']
                rgb = rgb.cpu()[j].permute(1, 2, 0).clip(0, 1).numpy()
                #if args.use_tgt_bboxes_as_mask:
                #    plt.imsave(os.path.join(qual_dir, f'{count}-tgt-rgb-mask.png'), rgb[:, :, 3])
                #    rgb = rgb[:, :, :3]
                plt.imsave(os.path.join(qual_dir, f'{count}-tgt-rgb.png'), rgb)

            wavfile.write(os.path.join(qual_dir, f'{count}-input.wav'), args.sample_rate, wav_dict['input'][j].T)
            wavfile.write(os.path.join(qual_dir, f'{count}-tgt.wav'), args.sample_rate, wav_dict['tgt'][j].T)
            wavfile.write(os.path.join(qual_dir, f'{count}-pred.wav'), args.sample_rate, wav_dict['pred'][j].T)
            if 'tgt_pose' in batch:
                with open(os.path.join(qual_dir, f'{count}-tgt-pose.json'), 'w') as fo:
                    pose = batch['tgt_pose'].cpu()[j].numpy()
                    json.dump({'theta': float(np.rad2deg(pose[1])), 'distance': float(pose[0])}, fo)
            if 'speaker_pose' in batch:
                with open(os.path.join(qual_dir, f'{count}-speaker-pose.json'), 'w') as fo:
                    pose = batch['speaker_pose'].cpu()[j].numpy()
                    json.dump({'theta': float(np.rad2deg(pose[1])), 'distance': float(pose[0])}, fo)
            count += 1

        webify(qual_dir, version="v1")

        html_short_dir = args.model_dir.split("/")[-1]
        html_dir = os.path.join("qual_htmls", html_short_dir)
        os.makedirs(html_dir, exist_ok=True)
        target_file = f'{html_dir}/v1.html'


        if os.path.exists(target_file):
            os.remove(target_file)
        os.symlink(os.path.join(os.path.abspath(qual_dir), 'index.html'), target_file)
        if count > 31:
            break
