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
import tqdm
import random

from novas.datasets.ss_speech_dataset import to_tensor
from novas.datasets.ss_speech_dataset import SoundSpacesSpeechDataset
from novas.models.base_av_model import load_rt60_estimator


class AVSpeechDataset(SoundSpacesSpeechDataset):
        def __init__(self, split, normalize_whole=True, normalize_segment=False, use_real_imag=False,
                    use_rgb=False, use_depth=False, limited_fov=False,
                    remove_oov=False, hop_length=160, use_librispeech=False, convolve_random_rir=False, use_da=False,
                    read_mp4=False, deterministic_eval=False, use_recv_audio=False, use_dereverb_sspeech=False, use_frozen_dereverb=False,
                    use_metricgan_dereverb=False, use_metricgan_debias=False, version="v1", static_rt60_eval=False, single_source_eval=False,
                    use_frozen_mgan_dvb=False, use_high_av=False):

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
            self.version = version
            self.static_rt60_eval = static_rt60_eval
            self.single_source_eval = single_source_eval
            self.use_frozen_mgan_dvb = use_frozen_mgan_dvb
            self.use_high_av = use_high_av

            self.data_dir = 'data/acoustic_avspeech/'
            self.img_dir = "high_av_img/" if self.use_high_av else "img/"
            self.audio_dir = "high_av_audio/" if self.use_high_av else "audio/"
            files = sorted(os.listdir(os.path.join(self.data_dir, self.img_dir)))
            files = [str(i) + '.png' for i in range(len(files))]

            if self.use_frozen_mgan_dvb:
                self.dbs_dir = "high_av_dbs" if self.use_high_av else "avspeech_dbs"
                audio_files = sorted(os.listdir(os.path.join(self.data_dir, self.dbs_dir, 'audio')))
                audio_idxs = [str(i.split(".wav")[0]) for i in audio_files]
                removed = 0
                for i in files:
                    if i.split(".png")[0] not in audio_idxs:
                        files.remove(i)
                        removed += 1
                print("removed " + str(removed) + " files")

            rt60_file_name = self.split + "_rt60.npy"
            rt60_scores_path = os.path.join(self.data_dir, rt60_file_name)
            #with open(rt60_scores_path, 'rb') as f:
            #    self.rt60_scores = np.load(f)

            
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
                if self.use_librispeech and self.use_high_av:
                    self.files = self.files[-1911:]

            #self.rt60_estimator = load_rt60_estimator(torch.device("cpu"))

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

            if self.use_metricgan_dereverb or self.use_metricgan_debias:
                fd = "acoustic_avspeech"
                if self.use_metricgan_dereverb:
                    self.mgan_path = os.path.join("data", fd, "metric_dereverb")
                if self.use_metricgan_debias:
                    self.mgan_path = os.path.join("data", fd, "metric_debias")
                audio_folder_name = "audio"
                if "test" in self.split and self.version != "v1":
                    print("using data generated by fine-tuned model...")
                    self.mgan_path = os.path.join(self.mgan_path, "dynamic_rt60/" + audio_folder_name)
                    if self.split == "test-seen":
                        self.mgan_path = os.path.join(self.mgan_path, "seen_audio")
                elif "test" in self.split and self.version == "v1" and self.static_rt60_eval:
                    print("using data generated by static model...")
                    self.mgan_path = os.path.join(self.mgan_path, "rt60/" + audio_folder_name)
                else:
                    print("using data generated by SRMR model...")
                    self.mgan_path = os.path.join(self.mgan_path, audio_folder_name)
                
                audio_files = sorted(os.listdir(self.mgan_path))
                audio_idxs = [str(i.split(".wav")[0]) for i in audio_files]
                removed = 0
                for i in self.files:
                    if i.split(".png")[0] not in audio_idxs:
                        self.files.remove(i)
                        removed += 1
                print("removed " + str(removed) + " files")
            
            if self.single_source_eval:
                self.wav_index = 52862

        def __len__(self):
            return len(self.files)
        
        def estimate_rt60(self, estimator, wav):
            stft = torch.stft(torch.from_numpy(wav), n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=torch.device("cpu")),
                            pad_mode='constant', return_complex=True)
            spec = torch.log1p(stft.abs()).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                estimated_rt60 = estimator(spec)
            return estimated_rt60
        
        def compute_weights(self):
            if self.split == "train":
                return self.rt60_scores
            
            train_rt60_path = os.path.join(self.data_dir, "train_rt60.npy")
            with open(train_rt60_path, 'rb') as f:
                train_rt60_scores = np.load(f)
            return train_rt60_scores
        
        def compute_rt60(self, wav):
            return self.estimate_rt60(self.rt60_estimator, wav).item()


        def __getitem__(self, item):
            if self.single_source_eval:
                img_file = os.path.join(self.data_dir, self.img_dir, str(self.wav_index) + ".png")
            else:
                img_file = os.path.join(self.data_dir, self.img_dir, self.files[item])
            audio_file = img_file.replace(self.img_dir, self.audio_dir).replace('.png', '.wav')
            rgb = np.array(Image.open(img_file))
            sr, recv_audio = wavfile.read(audio_file)
            if self.use_metricgan_dereverb or self.use_metricgan_debias:
                if self.single_source_eval:
                    mgan_file_name = self.files[self.wav_index].split(".png")[0] + ".wav"
                else:
                    mgan_file_name = self.files[item].split(".png")[0] + ".wav"
                sr, source_audio = wavfile.read(os.path.join(self.mgan_path, mgan_file_name))
                src_audio = np.array(source_audio)
            elif self.use_frozen_mgan_dvb:
                if self.single_source_eval:
                    avspeech_dbs_file = self.files[self.wav_index].replace('.png', '.wav')
                else:
                    avspeech_dbs_file = self.files[item].replace('.png', '.wav')
                sr, source_audio = wavfile.read(os.path.join(self.data_dir, self.dbs_dir + "/audio", avspeech_dbs_file))
                src_audio = np.array(source_audio)
            else:
                src_audio = np.zeros_like(recv_audio)
            id = int(self.files[item].split(".png")[0])

            if self.use_librispeech:
                if self.single_source_eval:
                    manual = 53867
                    try:
                        item_ind = self.files.index(str(manual) + ".png")
                    except ValueError:
                        item_ind = manual
                    ind = item_ind % len(self.speech_files)
                else:
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

            #sample['tgt_rt60'] = self.rt60_scores[item]
            
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
                rir = self.rir_list[item % len(self.rir_list)]
                padded_rir = np.pad(rir, (0, max(0, max_rir_len - rir.shape[0])))
                sample['rir'] = to_tensor(padded_rir)

            return sample
