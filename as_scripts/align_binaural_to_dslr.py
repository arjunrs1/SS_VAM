import os

import moviepy.editor as mp
import numpy as np
import math
from PIL import Image
from scipy.io import wavfile
import librosa
from scipy import signal

from temporal_align import gcc_phat


def main(scene_dir, video_index):
    scene_id = scene_dir.split('/')[-1]
    dslr_video = os.path.join(scene_dir, f'{scene_id}_DSLR_{video_index}.MP4')
    audio_fps = 48000

    # Read audio from mpeg video file:
    audioclip = mp.AudioFileClip(dslr_video)
    wav = audioclip.to_soundarray(fps=audio_fps)
    wav_data = np.array(wav)
    dslr_audio = wav_data.mean(axis=1)

    binaural_audio_left, _ = librosa.load(os.path.join(scene_dir, f'{scene_id}_BNRL_{video_index}A.WAV'), sr=None)
    binaural_audio_right, _ = librosa.load(os.path.join(scene_dir, f'{scene_id}_BNRL_{video_index}B.WAV'), sr=None)
    binaural_audio = np.array([binaural_audio_left, binaural_audio_right])
    wavfile.write('/tmp/tmp.wav', audio_fps, binaural_audio.T)

    binaural_clip = mp.AudioFileClip('/tmp/tmp.wav')

    lags = [0]
    lag, _ = gcc_phat(np.mean(binaural_audio, axis=0)[:30 * audio_fps], dslr_audio[:30 * audio_fps], fs=audio_fps)
    print(lag)
    lags.append(lag)
    lags = np.array(lags) - min(lags)

    video = mp.VideoFileClip(dslr_video, audio_fps=48000)
    video = video.cutout(0, lags[0])

    binaural_clip = binaural_clip.cutout(0, lags[1])

    video_subclip = video.subclip(t_start=0, t_end=400).without_audio()
    video_subclip.audio = binaural_clip.subclip(t_start=0, t_end=400)
    video_subclip.write_videofile('test.mp4', audio_fps=audio_fps, codec='libx264')


if __name__ == '__main__':
    main('data/appen/Scenes_Source/SC-1022', video_index=1)
