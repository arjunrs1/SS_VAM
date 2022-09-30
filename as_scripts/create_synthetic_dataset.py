import os
from concurrent.futures import ProcessPoolExecutor
import random
import glob
import pickle
import csv
import argparse
from PIL import Image
from tqdm import tqdm
import json
import shutil

import quaternion as qt
import habitat_sim
import numpy as np
from as_scripts.synthetic_dataset_utils import render_rgb, CLEAN_SCENE_SPLITS, \
    compute_speaker_identity_mapping, make_configuration, render_audio, compute_bboxes


def read_speech_files(split):
    random.seed(0)
    cvs_file_name = {'train': 'train-clean-360.csv', 'val': 'dev-clean.csv', 'val-seen': 'dev-clean.csv',
                     'test-seen': 'test-clean.csv', 'test': 'test-clean.csv'
                     }[split]
    with open(os.path.join('data/sounds/speech/LibriSpeech', cvs_file_name), newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        # ID, duration, wav, spk_id, wrd
        speech_files = [_ for _ in reader][1:]
    return speech_files


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def reachable_between_two_points(sim, pa, pb):
    if not sim.pathfinder.is_navigable(pa) or not sim.pathfinder.is_navigable(pb):
        return False

    euclidean_distance = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[2] - pb[2]) ** 2)
    path = habitat_sim.ShortestPath()
    path.requested_start = pa
    path.requested_end = pb
    sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance

    ratio = geodesic_distance / euclidean_distance

    if ratio < 1.2:
        return True
    else:
        return False


def equally_split_gender(speaker_identity_mapping, speech_files):
    f_files = []
    m_files = []
    for speech_file in speech_files:
        speech_id = speech_file[0].split(',')[0]
        gender = speaker_identity_mapping[int(speech_id.split('-')[0])]
        if gender == 'M':
            m_files.append(speech_file)
        else:
            f_files.append(speech_file)

    if len(f_files) == 0 or len(m_files) == 0:
        return None

    balanced_files = []
    f_idx = m_idx = 0
    for i in range(len(speech_files)):
        if i % 2 == 0:
            balanced_files.append(f_files[f_idx])
            f_idx = (f_idx + 1) % len(f_files)
        else:
            balanced_files.append(m_files[m_idx])
            m_idx = (m_idx + 1) % len(m_files)

    return balanced_files


def encode(scene):
    return abs(hash(scene)) % (10 ** 8)


def render_scene(scene, split, speech_file_indices, args, render_pano=False):
    seed = encode(scene)
    np.random.seed(seed)
    random.seed(seed)

    if render_pano:
        cfg = make_configuration('data', scene, resolution=(384, 64), fov=20)  # pano
    else:
        cfg = make_configuration('data', scene, resolution=(256, 256), fov=120)
    sim = habitat_sim.Simulator(cfg)
    # sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/mp3d/{scene}/{scene}.navmesh"))
    sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/gibson/Oyens.navmesh"))
    assert sim.pathfinder.is_loaded
    sim.pathfinder.seed(seed)

    speaker_identity_mapping = compute_speaker_identity_mapping()
    speech_files = read_speech_files(split)[speech_file_indices[0]:speech_file_indices[1]]
    speech_files = equally_split_gender(speaker_identity_mapping, speech_files)
    if speech_files is None:
        sim.close()
        return()

    speech_file_index = 0
    all_metadata = list()
    sample_count = 0
    pbar = tqdm(total=len(speech_files))
    # predefined_point = [2.75412, 0.02, 1.29692]
    while True:
        # num_speaker = np.random.randint(2, args.max_speaker+1)
        # num_active_speaker = np.random.randint(1, num_speaker+1)
        num_speaker = 2
        num_active_speaker = 2
        speaker_locations = [sim.pathfinder.get_random_navigable_point()]
        max_viewpoint = 4
        fail_count = 0
        while fail_count < 50:
            speaker_loc = speaker_locations[0] + [np.random.random() * args.radius, 0,
                                                  np.random.random() * args.radius]
            if reachable_between_two_points(sim, speaker_locations[0], speaker_loc) \
                    and dist(speaker_locations[0], speaker_loc) > 0.3:
                speaker_locations.append(speaker_loc)
                if len(speaker_locations) == num_speaker:
                    break
            else:
                fail_count += 1

        if fail_count == 50:
            continue

        center_location = np.sum(speaker_locations, axis=0) / len(speaker_locations)
        # TODO: for the target data, simulate facing directions away from the center
        # TODO: decide viewpoint based on whether there is direct sound
        # TODO: make sure the source view captures all speakers
        # TODO: check the fov, needless for the image to square
        # TODO: varying human meshes
        # TODO: nicer mesh environment (gibson?)
        # TODO: source view does not have to look at the center of the speakers
        viewpoint_locations = []
        viewpoint_rotations = []
        fail_count = 0
        while fail_count < 50:
            angle = np.random.random() * np.pi * 2
            distance = np.random.random() * 1.5 + 1
            viewpoint_loc = center_location - np.array([np.cos(angle) * distance, 0, np.sin(angle) * distance])
            if reachable_between_two_points(sim, viewpoint_loc, center_location):
                viewpoint_locations.append(viewpoint_loc)
                viewpoint_rotations.append(np.deg2rad(270) - angle)
                if len(viewpoint_locations) == max_viewpoint:
                    break
            else:
                fail_count += 1

        if len(viewpoint_locations) < 2:
            continue

        speaker_list = list()
        for k in range(num_speaker):
            if k < num_active_speaker:
                speech_id = speech_files[speech_file_index][0].split(',')[0]
                speech_file_index += 1
                pbar.update(1)
                active = True
                gender = speaker_identity_mapping[int(speech_id.split('-')[0])]
            else:
                speech_id = None
                active = False
                gender = np.random.choice(['M', 'F'])
            speaker_list.append({
                'id': k,
                'location': speaker_locations[k].tolist(),
                'gender': gender,
                'active': active,
                'speech': speech_id
            })

        viewpoint_list = list()
        for k in range(len(viewpoint_locations)):
            viewpoint_list.append({
                'id': k,
                'location': viewpoint_locations[k].tolist(),
                'rotation': viewpoint_rotations[k],
                'rir': {
                    spk['id']: os.path.join(args.data_dir, split, scene, str(sample_count), f"{spk['id']}_{k}.wav"
                                            ) for spk in speaker_list
                },
                'mono_rir': {
                    spk['id']: os.path.join(args.data_dir, split, scene, str(sample_count), f"mono_{spk['id']}_{k}.wav"
                                            ) for spk in speaker_list
                },
                'image': os.path.join(args.data_dir, split, scene, str(sample_count), f'{k}.png')
            })

        metadata = {'id': sample_count,
                    'viewpoints': viewpoint_list,
                    'speakers': speaker_list
                    }
        os.makedirs(os.path.join(args.data_dir, split, scene, str(sample_count)), exist_ok=True)
        # metadata = compute_bboxes(sim, metadata)
        render_rgb(sim, metadata, render_pano=render_pano)
        all_metadata.append(metadata)

        sample_count += 1
        # print(f'Current sample count: {sample_count}, speech file count: {speech_file_index}')
        if speech_file_index + args.max_speaker > len(speech_files):
            break

    pbar.close()
    sim.close()

    render_audio(scene, all_metadata, 'binaural')
    render_audio(scene, all_metadata, 'mono')

    with open(os.path.join(args.data_dir, split, scene, 'metadata.json'), 'w') as fo:
        print(f'Number of samples in split {split} scene {scene}: {len(all_metadata)}')
        json.dump(all_metadata, fo)


def main():
    """
    v2: 48K sample rate
    v3: 48K no indirect sound
    v4: same environment no indirect sound
    v5: same environment with indirect sound
    v6: same room with no indirect sound but varying S/R locations
    v7: 16K no indirect sound
    v8: 16K one male and one female speaker no indirect sound
    v9: 16K one male and one female speaker with indirect sound (rerun)
    v10: 16K one male and one female speaker no indirect sound both mono and binaural
    v11: 16K one male and one female speaker with indirect sound both mono and binaural, fixed height, new binaries
    v12: 16K one male and one female speaker no indirect sound both mono and binaural, fixed height, new binaries, no materials
    v13: v12 + indirect sound
    v14: v13 + materials (with config file)
    v15: single room
    v16: single room + fixed location sampling
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='data/synthetic_dataset/v16', type=str)
    parser.add_argument("--partition", default='learnfair,learnlab', type=str)
    parser.add_argument("--slurm", default=False, action='store_true')
    parser.add_argument("--max-speaker", default=3, type=int)
    parser.add_argument("--distance-limit", default=3, type=int)
    parser.add_argument("--radius", default=3, type=int)
    parser.add_argument("--overwrite", default=False, action='store_true')
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)

    scenes = list()
    splits = list()
    speech_file_indices = list()
    for split in ['train', 'val', 'test']:
        split_scenes = {'train': CLEAN_SCENE_SPLITS['train'],
                        'val': CLEAN_SCENE_SPLITS['val'],
                        'val-seen': CLEAN_SCENE_SPLITS['train'],
                        'test-seen': CLEAN_SCENE_SPLITS['train'],
                        'test': CLEAN_SCENE_SPLITS['test']
                        }[split]

        speech_files = read_speech_files(split)
        max_sample = int(len(speech_files) / len(split_scenes)) // 2
        print(f'Max speech files per scene in split {split} is {max_sample}')
        for i in range(len(split_scenes)):
            speech_file_indices.append((i * max_sample, (i+1) * max_sample))
        scenes += split_scenes
        splits += [split] * len(split_scenes)
    if args.slurm:
        import submitit
        executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
        executor.update_parameters(timeout_min=60*8, slurm_partition=args.partition, gpus_per_node=1, cpus_per_task=10)
        jobs = []
        num_machine = 100
        step = int(len(scenes) / num_machine) + 1
        with executor.batch():
            for i in range(num_machine):
                job = executor.submit(run_batch, scenes[i*step:(i+1)*step], splits[i*step:(i+1)*step],
                                      speech_file_indices[i*step:(i+1)*step], args)
                jobs.append(job)
    else:
        # with ProcessPoolExecutor() as executor:
        #     num_process = 10
        #     step = int(len(scenes) / num_process) + 1
        #     for i in range(num_process):
        #         job = executor.submit(run_batch, scenes[i * step:(i + 1) * step], splits[i * step:(i + 1) * step],
        #                               speech_file_indices[i * step:(i + 1) * step], args)
        run_batch(scenes, splits, speech_file_indices, args)


def run_batch(scenes, splits, speech_file_indices, args):
    for scene, file_indices, split in tqdm(zip(scenes, speech_file_indices, splits)):
        render_scene(scene, split, file_indices, args)


if __name__ == '__main__':
    main()
