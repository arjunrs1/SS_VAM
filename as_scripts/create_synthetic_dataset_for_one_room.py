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
from scipy.ndimage import measurements
from as_scripts.synthetic_dataset_utils import render_rgb, CLEAN_SCENE_SPLITS, \
    compute_speaker_identity_mapping, make_configuration, render_audio


def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


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


def render_scene(scene, split, speech_file_indices, args):
    cfg = make_configuration('data', scene, resolution=(256, 256), fov=120)
    sim = habitat_sim.Simulator(cfg)
    np.random.seed(0)
    random.seed(0)
    sim.seed(0)
    sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/mp3d/{scene}/{scene}.navmesh"))
    assert sim.pathfinder.is_loaded
    sim.pathfinder.seed(0)

    speaker_identity_mapping = compute_speaker_identity_mapping()
    graph = load_metadata(os.path.join('data/metadata/mp3d', scene))[1]
    speech_files = read_speech_files(split)[speech_file_indices[0]:speech_file_indices[1]]

    speech_file_index = 0
    all_metadata = list()
    sample_count = 0
    pbar = tqdm(total=len(speech_files))
    while True:
        if len(all_metadata) == 0:
            num_speaker = 2
            num_active_speaker = 2
            speaker_locations = [sim.pathfinder.get_random_navigable_point()]
            max_viewpoint = 2
            fail_count = 0
            while fail_count < 50:
                speaker_loc = speaker_locations[0] + [np.random.random() * args.radius, 0,
                                                      np.random.random() * args.radius]
                if reachable_between_two_points(sim, speaker_locations[0], speaker_loc):
                    speaker_locations.append(speaker_loc)
                    if len(speaker_locations) == num_speaker:
                        break
                else:
                    fail_count += 1

            if fail_count == 50:
                continue

            center_location = np.sum(speaker_locations, axis=0) / len(speaker_locations)
            # TODO: keep a minimum distance between viewpoints
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
        else:
            speaker_locations = [np.array(all_metadata[0]['speakers'][0]['location']) +
                                 np.array([np.random.random(), 0, np.random.random()]),
                                 np.array(all_metadata[0]['speakers'][1]['location']) +
                                 np.array([np.random.random(), 0, np.random.random()])]

            viewpoint_locations = [np.array(all_metadata[0]['viewpoints'][0]['location']) +
                                   np.array([np.random.random() * 0.3, 0, np.random.random() * 0.3]),
                                   np.array(all_metadata[0]['viewpoints'][1]['location']) +
                                   np.array([np.random.random() * 0.3, 0, np.random.random() * 0.3])]

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
                'speech': speech_id,
                # 'bbox': candidate_visible_locations_mapping[speaker_locations[k]] # TODO: add bbox
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
                'image': os.path.join(args.data_dir, split, scene, str(sample_count), f'{k}.png')
            })

        metadata = {'id': sample_count,
                    'viewpoints': viewpoint_list,
                    'speakers': speaker_list
                    }
        all_metadata.append(metadata)
        os.makedirs(os.path.join(args.data_dir, split, scene, str(sample_count)), exist_ok=True)
        render_rgb(sim, graph, metadata)

        sample_count += 1
        # print(f'Current sample count: {sample_count}, speech file count: {speech_file_index}')
        if speech_file_index + args.max_speaker > len(speech_files):
            break

    pbar.close()
    sim.close()

    render_audio(scene, all_metadata)

    with open(os.path.join(args.data_dir, split, scene, 'metadata.json'), 'w') as fo:
        print(f'Number of samples in split {split} scene {scene}: {len(all_metadata)}')
        json.dump(all_metadata, fo)

    print(all_metadata[0])


def compute_bbox(semantic):
    labeled_array, num_feat = measurements.label(semantic == 41)
    bboxes = []
    areas = []
    for i in range(num_feat):
        y_indices, x_indices = np.nonzero(labeled_array == i + 1)
        y_min = min(y_indices)
        y_max = max(y_indices)
        x_min = min(x_indices)
        x_max = max(x_indices)
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        if area < 50:
            continue
        bboxes.append((int(y_min), int(y_max), int(x_min), int(x_max)))
        areas.append(area)

    if len(areas) == 0:
        return None
    else:
        return bboxes[np.argmax(areas)]


def main():
    """
    v2: 48K sample rate
    v3: 48K no indirect sound
    v4: same environment no indirect sound
    v5: same environment with indirect sound
    v6: same room with no indirect sound but varying S/R locations
    :return:
    """
    np.random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default='data/synthetic_dataset/v6', type=str)
    parser.add_argument("--partition", default='learnfair,learnlab', type=str)
    parser.add_argument("--slurm", default=False, action='store_true')
    parser.add_argument("--max-speaker", default=3, type=int)
    parser.add_argument("--distance-limit", default=3, type=int)
    parser.add_argument("--radius", default=3, type=int)
    parser.add_argument("--overwrite", default=False, action='store_true')
    args = parser.parse_args()

    if args.overwrite and os.path.exists(args.data_dir):
        shutil.rmtree(args.data_dir)

    scene = '17DRP5sb8fy'
    for split in ['train', 'val', 'test']:
        max_sample = 2000 if split == 'train' else 500
        print(f'Max speech files per scene in split {split} is {max_sample}')
        file_indices = (0, max_sample)

        render_scene(scene, split, file_indices, args)


if __name__ == '__main__':
    main()
