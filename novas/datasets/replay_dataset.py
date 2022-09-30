# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
import torchvision
from decord import VideoReader, AudioReader
from decord import cpu
import librosa
from torch.utils.data import Dataset
import julius

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter("ignore", UserWarning)


def to_tensor(v):
    import numpy as np

    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def read_clip(file, split, target_sr):
    try:
        vr = VideoReader(file, ctx=cpu(0))
        rgb = vr[np.random.randint(0, len(vr)) if split == 'train' else 0].asnumpy()
        audio, input_sr = librosa.load(file.replace('.mp4', '.wav'), sr=None, mono=False)
        if input_sr != target_sr:
            audio = julius.resample_frac(torch.tensor(audio), input_sr, target_sr).numpy()
    except Exception as e:
        print(f'Exception {e} happened for {file}')
        rgb = np.zeros((720, 1280, 3))
        audio = np.zeros((2, 48000 * 2))

    return rgb, audio


class ReplayDataset(Dataset):
    def __init__(self, split, args):
        super().__init__()
        self.split = split
        self.args = args
        self.rgb_res = (216, 384)
        self.audio_len = 40960

        data_dir = 'data/replay_dataset'
        scenes = os.listdir(data_dir)
        clip_dirs = []
        for scene in scenes:
            clip_dirs += [os.path.join(data_dir, scene, clip_dir)
                          for clip_dir in os.listdir(os.path.join(data_dir, scene))]

        clip_dirs = [clip_dir for clip_dir in clip_dirs if len(os.listdir(clip_dir)) == 16]

        if split == 'train':
            self.clip_dirs = clip_dirs[: int(len(clip_dirs) * 0.95)]
        elif split == 'val':
            self.clip_dirs = clip_dirs[int(len(clip_dirs) * 0.95) + 1: int(len(clip_dirs) * 0.975)]
        else:
            self.clip_dirs = clip_dirs[int(len(clip_dirs) * 0.975) + 1:]

        print(f'Number of clip is {len(self.clip_dirs)} for {self.split.upper()}')

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, item):
        clip_dir = self.clip_dirs[item]

        file_1, file_2 = np.random.permutation([f for f in os.listdir(clip_dir) if f.endswith('.mp4')])[:2]
        file_1, file_2 = os.path.join(clip_dir, file_1), os.path.join(clip_dir, file_2)
        v1, a1 = read_clip(file_1, self.split, self.args.sample_rate)
        v2, a2 = read_clip(file_2, self.split, self.args.sample_rate)

        sample = dict()
        v1, v2 = self.process_rgb(v1.astype(np.float32)), self.process_rgb(v2.astype(np.float32))
        a1, a2 = self.process_audio(a1), self.process_audio(a2)
        random_start = np.random.randint(a1.shape[1] - self.audio_len)
        sample['src_wav'] = a1[:, random_start: random_start + self.audio_len]
        sample['tgt_wav'] = a2[:, random_start: random_start + self.audio_len]
        sample['src_rgb'] = to_tensor(v1).permute(2, 0, 1) / 255.0
        sample['tgt_rgb'] = to_tensor(v2).permute(2, 0, 1) / 255.0

        return sample

    def process_rgb(self, rgb):
        if rgb.shape[:2] != self.rgb_res:
            rgb = torchvision.transforms.Resize(self.rgb_res)(to_tensor(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()

        return rgb

    def process_audio(self, audio):
        return audio
