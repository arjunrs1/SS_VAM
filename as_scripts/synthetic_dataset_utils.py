import math
import os
from PIL import Image

import magnum as mn
import numpy as np
from scipy.ndimage import measurements

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import matplotlib
from scipy.io import wavfile

matplotlib.rcParams["figure.dpi"] = 600
SPEAKER_SEMANTIC_ID = 10000

# 3 environments removed from train and 3 removed from test
CLEAN_SCENE_SPLITS = {
    'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
              's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
              'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', '5LpN3gDmAk7',
              'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
              'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
              'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
              '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
              'uNb9QFRL6hY', 'ZMojNkEp431', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
              'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
              '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
    'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
            'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
    'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
             'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq',
             'YFuZgdQ5vWj', 'jtcxE69GiFV'],
}


def render_rgb(sim, metadata, render_pano=False):
    obj_templates_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # Load the object template from the configuration file
    obj_templates_mgr.load_configs(str(os.path.join("data/objects")))

    # Insert the object relative to the agent
    object_ids = []
    for speaker in metadata['speakers']:
        angle = np.random.random() * 360
        object_orientation = mn.Quaternion.rotation(mn.Deg(angle), mn.Vector3.y_axis())
        if speaker['gender'] == 'M':
            object_template_handle = obj_templates_mgr.get_template_handles("data/objects/male")[0]
        else:
            object_template_handle = obj_templates_mgr.get_template_handles("data/objects/female")[0]

        # Scale the object to fit the scene
        scaled_object_template = obj_templates_mgr.get_template_by_handle(object_template_handle)
        scaled_object_template.scale = np.array([0.8, 0.8, 0.8])
        obj_templates_mgr.register_template(scaled_object_template, "scaled")
        object = rigid_obj_mgr.add_object_by_template_handle("scaled")
        object.translation = np.array(speaker['location']) + np.array([0, 0.8, 0])
        object.rotation = object_orientation
        object_ids.append(object.object_id)

    if render_pano:
        angles = [0, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20]
        for viewpoint in metadata['viewpoints']:
            rgb_panorama = []
            depth_panorama = []
            for angle in angles:
                agent_state = habitat_sim.AgentState()
                agent_state.position = viewpoint['location']
                agent_state.rotation = quat_from_angle_axis(
                    math.radians(angle), np.array([0, 1.0, 0])
                ) * quat_from_angle_axis(math.radians(0), np.array([1.0, 0, 0]))
                agent = sim.initialize_agent(0, agent_state)

                observation = sim.get_sensor_observations()
                rgb_panorama.append(observation["rgba_camera"][..., :3])
                depth_panorama.append(normalize_depth(observation['depth_camera']) * 255.0)
            rgb_panorama = np.concatenate(rgb_panorama, axis=1)
            depth_panorama = np.concatenate(depth_panorama, axis=1)
            rgb = Image.fromarray(rgb_panorama.astype(np.uint8))
            rgb.save(viewpoint['image'][:-4] + '_pano.png')

            rgb = Image.fromarray(depth_panorama.astype(np.uint8))
            rgb.save(viewpoint['image'][:-4] + '_depth_pano.png')
    else:
        for viewpoint in metadata['viewpoints']:
            agent_state = habitat_sim.AgentState()
            agent_state.position = viewpoint['location']
            agent_state.rotation = quat_from_angle_axis(viewpoint['rotation'], np.array([0, 1.0, 0]))
            agent = sim.initialize_agent(0, agent_state)

            observation = sim.get_sensor_observations()
            rgb = observation["rgba_camera"][..., :3]
            rgb = Image.fromarray(rgb.astype(np.uint8))
            rgb.save(viewpoint['image'])

            depth = observation['depth_camera']
            rgb = Image.fromarray((normalize_depth(depth) * 255.0).astype(np.uint8))
            rgb.save(viewpoint['image'][:-4] + '_depth.png')

    rigid_obj_mgr.remove_all_objects()


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def compute_bboxes(sim, metadata):
    scene = sim.semantic_scene
    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id))])

    for viewpoint in metadata['viewpoints']:
        viewpoint['bboxes'] = {}
        agent_state = habitat_sim.AgentState()
        agent_state.position = viewpoint['location']
        agent_state.rotation = quat_from_angle_axis(viewpoint['rotation'], np.array([0, 1.0, 0]))
        agent = sim.initialize_agent(0, agent_state)

        obj_templates_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()
        obj_templates_mgr.load_configs(str(os.path.join("data/objects")))

        for speaker in metadata['speakers']:
            angle = 0
            object_orientation = mn.Quaternion.rotation(mn.Deg(angle), mn.Vector3.y_axis())
            if speaker['gender'] == 'M':
                object_template_handle = obj_templates_mgr.get_template_handles("data/objects/male")[0]
            else:
                object_template_handle = obj_templates_mgr.get_template_handles("data/objects/female")[0]

            # Scale the object to fit the scene
            scaled_object_template = obj_templates_mgr.get_template_by_handle(object_template_handle)
            scaled_object_template.scale = np.array([0.8, 0.8, 0.8])
            obj_templates_mgr.register_template(scaled_object_template, "scaled")
            object = rigid_obj_mgr.add_object_by_template_handle("scaled")
            object.semantic_id = SPEAKER_SEMANTIC_ID
            object.translation = np.array(speaker['location']) + np.array([0, 0.8, 0])
            object.rotation = object_orientation

            observation = sim.get_sensor_observations()
            rgb = observation["rgba_camera"][..., :3]
            seg = observation['semantic_camera']
            if SPEAKER_SEMANTIC_ID in seg:
                speaker_indices = seg == SPEAKER_SEMANTIC_ID
                seg[speaker_indices] = 0
                semantic = np.take(mapping, seg)
                assert 41 not in semantic
                semantic[speaker_indices] = 41
                bbox = semantic_to_bbox(semantic)
            else:
                bbox = None

            if bbox is not None:
                # import matplotlib.pyplot as plt
                # import matplotlib.patches as patches
                # fig, ax = plt.subplots()
                # plt.imshow(rgb)
                # y_min, y_max, x_min, x_max = np.array(bbox)
                # print(y_min, y_max, x_min, x_max)
                # ax.add_patch(
                #     patches.Rectangle(
                #         (x_min, y_min),
                #         x_max - x_min,
                #         y_max - y_min,
                #         fill=False,
                #         linewidth=2, edgecolor='r', facecolor='none'
                #     ))
                # plt.draw()
                # plt.savefig('rgb.png')
                # plt.imsave('semantic.png', semantic)
                # plt.close()

                viewpoint['bboxes'][speaker['id']] = bbox
            else:
                viewpoint['bboxes'][speaker['id']] = -1
            rigid_obj_mgr.remove_all_objects()

    return metadata


def semantic_to_bbox(semantic):
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


def add_acoustic_config(sim, mic):
    # create the acoustic configs
    # acoustics_config = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    # acoustics_config.enableMaterials = True
    # acoustics_config.sampleRate = 16000
    # acoustics_config.indirect = False
    # acoustics_config.transmission = False

    # # create channel layout
    # channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    # if mic == 'binaural':
    #     channel_layout.type = (
    #         habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    #     )
    #     channel_layout.channelCount = 2
    # elif mic == 'mono':
    #     channel_layout.type = (
    #         habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
    #     )
    #     channel_layout.channelCount = 1

    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = False
    audio_sensor_spec.acousticsConfig.sampleRate = 16000
    audio_sensor_spec.acousticsConfig.indirect = True
    audio_sensor_spec.acousticsConfig.transmission = False
    if mic == 'mono':
        audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono
        audio_sensor_spec.channelLayout.channelCount = 1
    elif mic == 'binaural':
        audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        audio_sensor_spec.channelLayout.channelCount = 2

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON('data/mp3d_material_config.json')


def render_audio(scene, all_metadata, mic='binaural'):
    cfg = make_configuration('data', scene)
    sim = habitat_sim.Simulator(cfg)
    add_acoustic_config(sim, mic)

    for metadata in all_metadata:
        for speaker in metadata['speakers']:
            audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
            audio_sensor.setAudioSourceTransform(np.array(speaker['location']) + np.array([0, 1.5, 0]))

            for viewpoint in metadata['viewpoints']:
                agent = sim.get_agent(0)
                new_state = sim.get_agent(0).get_state()
                new_state.position = viewpoint['location']
                new_state.rotation = quat_from_angle_axis(viewpoint['rotation'], np.array([0, 1, 0]))
                new_state.sensor_states = {}
                agent.set_state(new_state, True)

                wav = np.array(sim.get_sensor_observations()['audio_sensor'])
                if mic == 'binaural':
                    wavfile.write(viewpoint['rir'][speaker['id']], 16000, wav.T)
                else:
                    wavfile.write(viewpoint['mono_rir'][speaker['id']], 16000, wav.T)

    sim.close()


def compute_speaker_identity_mapping():
    mapping = {}
    with open('data/sounds/speech/LibriSpeech/SPEAKERS.TXT', 'r') as fo:
        lines = fo.readlines()[12:]
        for line in lines:
            tokens = [x for x in line.replace('|', '').split(' ') if x]
            speaker_id = int(tokens[0])
            speaker_gender = tokens[1]
            assert speaker_gender == 'F' or speaker_gender == 'M'
            mapping[speaker_id] = speaker_gender

    return mapping


def make_configuration(data_path, scene, resolution=(512, 256), fov=20):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    # backend_cfg.scene_id = os.path.join(data_path, f"scene_datasets/mp3d/{scene}/{scene}.glb")
    # backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    backend_cfg.scene_id = 'data/scene_datasets/gibson/Oyens.glb'
    backend_cfg.enable_physics = False

    # agent configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.resolution = resolution
    rgb_sensor_cfg.far = np.iinfo(np.int32).max
    rgb_sensor_cfg.hfov = mn.Deg(fov)
    rgb_sensor_cfg.position = np.array([0, 1.3, 0])

    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = 'depth_camera'
    depth_sensor_cfg.resolution = resolution
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.hfov = mn.Deg(fov)
    depth_sensor_cfg.position = np.array([0, 1.3, 0])

    semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
    semantic_sensor_cfg.uuid = "semantic_camera"
    semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_cfg.resolution = resolution
    semantic_sensor_cfg.hfov = mn.Deg(fov)
    semantic_sensor_cfg.position = np.array([0, 1.3, 0])

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg, semantic_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])
