import os
import json
from concurrent.futures import ProcessPoolExecutor

import moviepy.editor as mp
import numpy as np
import submitit
from tqdm import tqdm
from scipy.io import wavfile
import librosa

from as_scripts.temporal_align import gcc_phat, START_TIME


def extract_scene(source_dir, output_dir, scene):
    clip_length = 3

    print(source_dir, output_dir, scene)
    scene_dir = os.path.join(source_dir, scene)
    os.makedirs(scene_dir, exist_ok=True)
    audio_fps = 48000

    metadata_file = os.path.join(f'data/replay_metadata/{scene}.json')
    if not os.path.exists(metadata_file):
        return
    with open(metadata_file, 'r') as fo:
        metadata = json.load(fo)
    valid_indices = metadata['valid_indices']
    dslr_lags = {int(key): value for key, value in metadata['dslr_lags'].items()}
    binaural_lags = {int(key): value for key, value in metadata['binaural_lags'].items()}
    duration = metadata['duration']

    if scene in START_TIME:
        index, start_time = START_TIME[scene]
    else:
        index = valid_indices[0]
        start_time = 210
    relative_start_time = start_time - binaural_lags[index]

    dslr_videos = {}
    for index in valid_indices:
        dslr_video_file = os.path.join(scene_dir, f'{scene}_DSLR_{index}.MP4')
        video = mp.VideoFileClip(dslr_video_file, audio_fps=audio_fps)
        video = video.cutout(0, dslr_lags[index])
        dslr_videos[index] = video

    binaural_audios = {}
    for index in valid_indices:
        binaural_left, sr = librosa.load(os.path.join(scene_dir, f'{scene}_BNRL_{index}A.WAV'), sr=audio_fps)
        binaural_right, sr = librosa.load(os.path.join(scene_dir, f'{scene}_BNRL_{index}B.WAV'), sr=audio_fps)
        binaural_audios[index] = np.array([binaural_left, binaural_right])[:, int(binaural_lags[index] * audio_fps):]

    start_time = relative_start_time
    count = 0
    while start_time + clip_length < duration:
        for index in valid_indices:
            if start_time + clip_length > dslr_videos[index].duration:
                break
            subclip = dslr_videos[index].subclip(start_time, start_time + clip_length).set_start(0).resize(0.2)
            os.makedirs(os.path.join(output_dir, scene, str(count)), exist_ok=True)
            subclip.write_videofile(os.path.join(output_dir, scene, str(count), f'{index}.mp4'),
                                    audio_fps=audio_fps, codec='libx264',
                                    temp_audiofile=f'/tmp/{np.random.randint(1000000000)}.mp3')
            wavfile.write(os.path.join(output_dir, scene, str(count), f'{index}.wav'), audio_fps, binaural_audios[index]
                          [:, int(start_time * audio_fps): int((start_time+clip_length) * audio_fps)].T)

        start_time += clip_length
        count += 1


def concurrent_extract_scene(source_dir, output_dir, scenes):
    # print(source_dir, output_dir, scenes)
    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     for scene in scenes:
    #         executor.submit(extract_scene, source_dir, output_dir, scene)
    for scene in os.listdir(source_dir):
        extract_scene(source_dir, output_dir, scene)


def main():
    source_dir = 'data/appen/Scenes_Source'
    output_dir = 'data/replay_dataset'
    # scenes = [x.split('.')[0] for x in os.listdir('data/replay_metadata')]
    scenes = ['SC-1105']

    for scene in scenes:
        extract_scene(source_dir, output_dir, scene)

    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     for scene in os.listdir(source_dir):
    #         executor.submit(extract_scene, source_dir, output_dir, scene)

    # import submitit
    #
    # executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
    # executor.update_parameters(slurm_job_name='create_dataset', timeout_min=60 * 12,
    #                            slurm_partition='learnlab,learnfair', nodes=1, gpus_per_node=1, cpus_per_task=10,
    #                            slurm_constraint='volta32gb' if False else None, slurm_mem=100 * 1024,
    #                            tasks_per_node=1, comment=''
    #                            )
    #
    # with executor.batch():
    #     for scene in scenes:
    #         executor.submit(extract_scene, source_dir, output_dir, scene)


if __name__ == '__main__':
    main()
