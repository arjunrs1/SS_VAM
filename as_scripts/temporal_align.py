import os

import moviepy.editor as mp
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import librosa
from scipy import signal


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16, return_second=True):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    if not return_second:
        tau *= fs

    return tau, cc


START_TIME = {
    'SC-1022': (1, 165),
    'SC-1023': (1, 210),
    'SC-1024': (1, 180),
    'SC-1025': (1, 110),
    'SC-1026': (1, 185),
    'SC-1027': (1, 170),
    'SC-1028': (3, 170),
    'SC-1032': (1, 180),
    'SC-1033': (1, 170),
    'SC-1035': (1, 160),
    'SC-1036': (1, 180),
    'SC-1037': (1, 250),
    'SC-1039': (1, 205),
    'SC-1040': (1, 230),
    'SC-1042': (1, 180),
    'SC-1043': (1, 140),
    'SC-1044': (1, 170),
    'SC-1045': (1, 160),
    'SC-1046': (1, 170),
    'SC-1047': (1, 180),
    'SC-1048': (1, 220),
    'SC-1049': (1, 170),
    'SC-1050': (1, 210),
    'SC-1052': (1, 140),
    'SC-1053': (1, 180),
    'SC-1054': (1, 140),
    # TODO: add the rest
}


SCENE_ANNOTATIONS = {
    # indices to skip, DSLR index and binaural index for computing correlation, seconds (default 120)
    'SC-1022': ([5], 1, 1),
    'SC-1023': ([5], 1, 1),
    'SC-1024': ([5], 1, 1),
    'SC-1025': ([], 1, 1),
    'SC-1026': ([], 1, 1),
    'SC-1027': ([], 1, 1),
    'SC-1028': ([], 4, 6),
    'SC-1032': ([], 2, 2),
    'SC-1033': ([], 7, 7),
    'SC-1035': ([], 6, 6),
    'SC-1036': ([], 4, 7),
    'SC-1037': ([], 4, 5),  # to be checked
    'SC-1038': ([], 4, 5),
    'SC-1039': ([], 5, 1),
    'SC-1040': ([], 8, 5),
    'SC-1042': ([], 6, 7),
    'SC-1043': ([], 7, 1),
    'SC-1044': ([], 2, 3),
    'SC-1045': ([], 1, 1),
    'SC-1046': ([], 2, 1),
    'SC-1047': ([], 1, 1),
    'SC-1048': ([], 1, 1),
    'SC-1049': ([], 6, 5, 180),
    'SC-1050': ([], 6, 7, 180),
    'SC-1052': ([], 2, 2),
    'SC-1053': ([], 6, 6),
    'SC-1054': ([], 7, 7),
    'SC-1059': ([1], 4, 4),
    'SC-1060': ([], 4, 4),
    'SC-1061': ([], 4, 4),
    'SC-1062': ([], 4, 4),
    'SC-1065': ([], 4, 4),
    'SC-1066': ([1], 6, 6),
    'SC-1069': ([1], 2, 2),
    'SC-1070': ([1], 5, 5),
    'SC-1071': ([1], 4, 4),
    'SC-1073': ([1], 5, 5),
    'SC-1074': ([1], 7, 7),
    'SC-1075': ([1], 5, 5),
    'SC-1076': ([1], 4, 4),
    'SC-1077': ([1], 4, 4),
    'SC-1078': ([1], 2, 2),
    'SC-1079': ([1], 4, 4),
    'SC-1080': ([], 4, 4),
    'SC-1081': ([1], 4, 4),
    'SC-1082': ([1], 5, 5),
    'SC-1084': ([1], 4, 4),
    'SC-1085': ([1], 4, 4),
    'SC-1086': ([1], 4, 4),
    'SC-1087': ([1], 4, 4),
    'SC-1088': ([1], 4, 4),
    'SC-1089': ([1], 4, 4),
    'SC-1090': ([1], 5, 5),
    'SC-1091': ([1], 5, 5),
    'SC-1092': ([1], 3, 3),
    'SC-1093': ([1], 6, 5),
    'SC-1099': ([1], 2, 4),
    'SC-1101': ([1], 4, 4),
    'SC-1102': ([1], 4, 4),
    'SC-1103': ([1], 6, 5),
    'SC-1104': ([1], 5, 5),
    'SC-1105': ([1], 3, 5),
    'SC-1106': ([1], 5, 5),
    'SC-1107': ([], 6, 5),
    'SC-1108': ([1], 3, 4),
    'SC-1109': ([1], 8, 8),
}


def return_valid_indices(scene_id):
    indices = list(range(1, 9))
    for idx in SCENE_ANNOTATIONS[scene_id][0]:
        indices.remove(idx)
    return indices


# noinspection DuplicatedCode
def main():
    """
    Issues:
    SC-1024 does not have SC-1024_BNRL_8B.WAV
    SC-1028-DSLR-1.mp4 invalid data while processing input (skipped)
    """

    scenes = sorted(os.listdir('data/appen/Scenes_Source/'))
    audio_fps = 16000

    for scene_id in tqdm(scenes):
        if int(scene_id.split('-')[-1]) not in [1037, 1045, 1075, 1086, 1089, 1099, 1103, 1105]:
            continue

        # if int(scene_id.split('-')[-1]) > 1023:
        #     break

        print(scene_id)
        scene_dir = f'data/appen/Scenes_Source/{scene_id}'

        # dslr_video_files = {}
        dslr_audios = {}
        binaural_audios = {}
        valid_indices = []
        for index in return_valid_indices(scene_id):
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
            dslr_audios[index] = dslr_audio
            # dslr_video_files[index] = dslr_video_file
            binaural_audios[index] = binaural_audio

        print(valid_indices)
        if len(valid_indices) < 2:
            continue

        max_len = 60 * 6 * audio_fps
        dslr_anchor_index = SCENE_ANNOTATIONS[scene_id][1]
        dslr_lags = {dslr_anchor_index: 0}
        for index in valid_indices:
            if index == dslr_anchor_index:
                continue
            lag, _ = gcc_phat(dslr_audios[index][:max_len],
                              dslr_audios[dslr_anchor_index][:max_len], fs=audio_fps)
            dslr_lags[index] = lag
        print(f'DSLR lags: {dslr_lags}')

        binaural_anchor_index = SCENE_ANNOTATIONS[scene_id][2]
        binaural_lags = {binaural_anchor_index: 0}
        for index in valid_indices:
            if index == binaural_anchor_index:
                continue
            lag, _ = gcc_phat(binaural_audios[index][:max_len],
                              binaural_audios[binaural_anchor_index][:max_len], fs=audio_fps)
            binaural_lags[index] = lag
        print(f'Binaural lags: {binaural_lags}')

        if len(SCENE_ANNOTATIONS[scene_id]) == 3:
            cutoff_time = 120
        else:
            cutoff_time = SCENE_ANNOTATIONS[scene_id][3]
        dslr_anchor_clap_time = np.argmax(np.abs(dslr_audios[dslr_anchor_index][:cutoff_time * audio_fps])) / audio_fps
        binaural_anchor_clap_time = np.argmax(np.abs(binaural_audios[binaural_anchor_index][:cutoff_time * audio_fps])) / audio_fps
        print(f'Clap times: {dslr_anchor_clap_time}, {binaural_anchor_clap_time}')

        clap_time_diff = binaural_anchor_clap_time - dslr_anchor_clap_time
        for index in binaural_lags:
            binaural_lags[index] += clap_time_diff

        min_lag = min(list(dslr_lags.values()) + list(binaural_lags.values()))
        print('Min lag: ', min_lag)
        for index in dslr_lags:
            dslr_lags[index] -= min_lag
        for index in binaural_lags:
            binaural_lags[index] -= min_lag

        min_duration = min([dslr_audios[index].shape[0] // audio_fps - dslr_lags[index] for index in valid_indices] +
                           [binaural_audios[index].shape[0] // audio_fps - binaural_lags[index]
                            for index in valid_indices])
        print(f'Min duration: {min_duration}')

        os.makedirs(f'data/verify_clap/{scene_id}', exist_ok=True)
        relative_clap_time = dslr_anchor_clap_time - dslr_lags[dslr_anchor_index]
        for index in valid_indices:
            dslr_start = int((dslr_lags[index] + relative_clap_time - 0.5) * audio_fps)
            binaural_start = int((binaural_lags[index] + relative_clap_time - 0.5) * audio_fps)
            wavfile.write(f'data/verify_clap/{scene_id}/{index}.wav', audio_fps,
                          dslr_audios[index][dslr_start: dslr_start + 3 * audio_fps])
            wavfile.write(f'data/verify_clap/{scene_id}/{index}_BNRL.wav', audio_fps,
                          binaural_audios[index][binaural_start: binaural_start + 3 * audio_fps])

        write_metadata(scene_id, valid_indices, dslr_lags, binaural_lags, min_duration)


def write_metadata(scene_id, valid_indices, dslr_lags, binaural_lags, min_duration):
    import json

    metadata_dir = 'data/replay_metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, f'{scene_id}.json'), 'w') as fo:
        json.dump({
            'valid_indices': valid_indices,
            'dslr_lags': dslr_lags,
            'binaural_lags': binaural_lags,
            'duration': min_duration,  # starting from start
        }, fo)


if __name__ == '__main__':
    main()
