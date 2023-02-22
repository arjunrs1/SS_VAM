#Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import pickle

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from PIL import Image
from scipy.io import wavfile

from novas.datasets.ss_speech_dataset import to_tensor
from novas.datasets.ss_speech_dataset import SoundSpacesSpeechDataset


class AVSpeechDataset(SoundSpacesSpeechDataset):
        def __init__(self, split, normalize_whole=True, normalize_segment=False, use_real_imag=False,
                    use_rgb=False, use_depth=False, limited_fov=False,
                    remove_oov=False, hop_length=160, use_librispeech=False, convolve_random_rir=False, use_da=False,
                    read_mp4=False, deterministic_eval=False, use_recv_audio=False, use_dereverb_sspeech=False, use_frozen_dereverb=False,
                    use_metricgan_dereverb=False, use_metricgan_debias=False):

            super().__init__(split)
            self.split = split
            self.normalize_whole = normalize_whole
            self.normalize_segment = normalize_segment
            self.use_real_imag = use_real_imag
            self.use_rgb = use_rgb
            self.use_depth = use_depth
            self.limited_fov = limited_fov
            self.hop_length = hop_length
            self.rgb_res = (192, 576) #180, 320
            self.use_librispeech = use_librispeech
            self.convolve_random_rir = convolve_random_rir
            self.use_da = use_da
            self.read_mp4 = read_mp4
            self.use_frozen_dereverb = use_frozen_dereverb
            self.use_metricgan_dereverb = use_metricgan_dereverb
            self.use_metricgan_debias = use_metricgan_debias

            self.data_dir = 'data/acoustic_avspeech/'
            files = sorted(os.listdir(os.path.join(self.data_dir, 'img')))
            files = [str(i) + '.png' for i in range(len(files))]

            #with open('data/acoustic_avspeech/missing.npy', 'rb') as f:
            #    missing = np.load(f)

            #missing = [str(m)+".png" for m in list(missing)]
            #train_missing_files = missing

            if split == 'train':
                self.files = files[: int(len(files) * 0.95)]
            elif split == 'test-seen':
                self.files = files[: int(len(files) * 0.025)]
            elif split == 'val':
                self.files = files[int(len(files) * 0.95) + 1: int(len(files) * 0.975)]
            else:
                self.files = files[int(len(files) * 0.975) + 1:]

            if use_librispeech:
                speech_split = split if split != 'test-seen' else 'train'
                self.speech_files = sorted(glob.glob(f'data/soundspaces_speech/{speech_split}/**/*.pkl', recursive=True))

            if self.convolve_random_rir:
                rir_dict_file = 'data/acoustic_avspeech/random_rir.pkl'
                assert os.path.exists(rir_dict_file)
                with open(rir_dict_file, 'rb') as fo:
                    rir_split = split if split != 'test-seen' else 'train'
                    self.rir_list = pickle.load(fo)[rir_split]
                    print(f'Number of rirs: {len(self.rir_list)}')
                    
            if use_da:
                # 270, 480, 144, 256
                #resize: height=270, width=480
                transforms = [A.Resize(height=192, width=576)] if self.read_mp4 else []
                if split == 'train':
                    transforms += [
                            #A.RandomCrop(height=180, width=320),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                        ]

                    
                else:
                    transforms += [
                            #A.CenterCrop(height=180, width=320),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                        ]
                self.transform = A.Compose(transforms)

        def __len__(self):
            return len(self.files)

        def __getitem__(self, item):
            img_file = os.path.join(self.data_dir, 'img', self.files[item])
            audio_file = img_file.replace('img/', 'audio/').replace('.png', '.wav')
            rgb = np.array(Image.open(img_file))
            sr, recv_audio = wavfile.read(audio_file)
            if self.use_metricgan_dereverb:
                devb_file = img_file.replace('img/', 'metric_dereverb/audio/').replace('.png', '.wav')
                sr, src_audio = wavfile.read(devb_file)
            elif self.use_metricgan_debias:
                debias_file = img_file.replace('img/', 'metric_debias/audio/').replace('.png', '.wav')
                sr, src_audio = wavfile.read(debias_file)
            else:
                src_audio = np.zeros_like(recv_audio)
            id = int(self.files[item].split(".png")[0])

            if self.use_librispeech:
                ind = item % len(self.speech_files)
                speech_file = self.speech_files[ind]
                with open(speech_file, 'rb') as fo:
                    speech_data = pickle.load(fo)
                src_audio = speech_data['source_audio']
            

            if rgb.shape[:2] != self.rgb_res:
                rgb = torchvision.transforms.Resize(self.rgb_res)(to_tensor(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            rgb = rgb.astype(np.float32)
            
            sample = dict()
            sample['id'] = id
            
            src_wav, recv_wav = self.process_audio(src_audio, recv_audio)
            
            #and test not in split - why did you add this in earlier?
            if self.use_frozen_dereverb:
                sample['src_wav'] = recv_wav
                sample['tgt_wav'] = recv_wav
            else:
                sample['src_wav'] = src_wav
                sample['tgt_wav'] = recv_wav
            
            if self.use_da:
                sample['original_rgb'] = to_tensor(rgb).permute(2, 0, 1) / 255.0
                sample['src_rgb'] = self.transform(image=rgb)['image']
            else:
                sample['src_rgb'] = to_tensor(rgb).permute(2, 0, 1) / 255.0
            if self.use_depth:
                sample['src_depth'] = torch.ones((1,sample['src_rgb'].shape[1], sample['src_rgb'].shape[2]))
            
            if self.convolve_random_rir:
                max_rir_len = 24000
                # rir = np.random.choice(self.rir_list)
                rir = self.rir_list[item % len(self.rir_list)]
                padded_rir = np.pad(rir, (0, max(0, max_rir_len - rir.shape[0])))
                sample['rir'] = to_tensor(padded_rir)

            return sample
