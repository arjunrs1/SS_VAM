import os

import moviepy.editor as mp
import numpy as np
from scipy import signal
from IPython.lib.display import Audio
from as_scripts.temporal_align import gcc_phat
import librosa
from librosa.display import waveshow
import matplotlib.pyplot as plt


scene_ids = os.listdir('data/appen/Scenes_Source')
print(scene_ids)
# scene_dir = '../data/appen/Scenes_Source/SC-1024'
for scene_id in scene_ids:
    scene_dir = os.path.join('data/appen/Scenes_Source', scene_id)
    if int(scene_id.split('-')[-1]) < 1022:
        continue
    audio_fps = 16000

    dslr_video_files = []
    dslr_audios = []
    binaural_audios = []
    full_binaural_audios = []
    valid_indices = []
    for index in range(1, 9):
        dslr_video_file = os.path.join(scene_dir, f'{scene_id}_DSLR_{index}.MP4')
        if not os.path.exists(dslr_video_file):
            continue

        try:
            dslr_audio = np.array(mp.AudioFileClip(dslr_video_file).to_soundarray(fps=audio_fps))
            if dslr_audio.shape[0] < audio_fps * 60:
                raise ValueError
            dslr_audio = dslr_audio.mean(axis=1)
            dslr_audio /= np.max(np.abs(dslr_audio))
            print(f'Video length: {dslr_audio.shape[0] // audio_fps}')

            binaural_left, sr = librosa.load(os.path.join(scene_dir, f'{scene_id}_BNRL_{index}A.WAV'), sr=audio_fps)
            binaural_right, sr = librosa.load(os.path.join(scene_dir, f'{scene_id}_BNRL_{index}B.WAV'), sr=audio_fps)
            binaural_audio = np.mean([binaural_left, binaural_right], axis=0)
            binaural_audio /= np.max(np.abs(binaural_audio))
            print(f'Binaural length: {binaural_audio.shape[0] // audio_fps}')

        except Exception as e:
            print(e)
            print(f'Skipping index {index}')
            continue

        valid_indices.append(index)
        dslr_audios.append(dslr_audio)
        dslr_video_files.append(dslr_video_file)
        binaural_audios.append(binaural_audio)

        fig, axes = plt.subplots(len(dslr_audios) * 2, 1, figsize=(6, 30))
        for i in range(len(dslr_audios)):
            axes[2 * i].set_title(f'Index {valid_indices[i]}')
            waveshow(dslr_audios[i], sr=audio_fps, ax=axes[2 * i])
            waveshow(binaural_audios[i], sr=audio_fps, ax=axes[2 * i + 1])
        plt.savefig(f'data/plot_waves/{scene_id}.png')
        plt.close(fig)
