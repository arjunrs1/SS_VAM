# -*- coding: utf-8 -*-

import os
import json
import string
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio
import tqdm
import torch
import novas
from novas.models.wavenet import WaveNet
import torchaudio
import pickle
from novas.datasets.avspeech_dataset import AVSpeechDataset
import numpy as np
dataset = AVSpeechDataset

logger = logging.getLogger(__name__)
LEXICON_URL = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
TRAIN_JSON = "train_revb.json"
TEST_JSON = "test_revb.json"
VALID_JSON = "valid_revb.json"
FULL_JSON = "full_revb.json"
SAMPLERATE = 16000

def prepare_data(
    data_folder, save_folder, skip_prep=False
):
    """
    Prepares the json files for the AVSpeech dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original AVSpeech dataset is stored.
    save_folder : str
        The directory where to store the json files.
    valid_speaker_count : int
        The number of validation speakers to use (out of 28 in train set).
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> data_folder = '/path/to/datasets/SoundSpaces'
    >>> save_folder = 'exp/SoundSpaces_exp'
    >>> prepare_data(data_folder, save_folder)
    """

    if skip_prep:
        return

    # Setting ouput files
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_valid = os.path.join(save_folder, VALID_JSON)
    save_json_test = os.path.join(save_folder, TEST_JSON)
    save_json_full = os.path.join(save_folder, FULL_JSON)


    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_test, save_json_valid, save_json_full):
        logger.info("Preparation completed in previous run, skipping.")
        return


    #noisy_folder = os.path.join(data_folder, "audio")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger.debug("Creating lexicon...")
    logger.info("Creating json files for SoundSpaces...")

    logger.debug("Collecting files...")
    extension = [".wav"]

    tgt_file = os.path.join(data_folder, 'remove_oov', 'train.pkl')
    with open(tgt_file, 'rb') as fo:
        wav_lst_train = pickle.load(fo)

    tgt_file = os.path.join(data_folder, 'remove_oov', 'val.pkl')
    with open(tgt_file, 'rb') as fo:
        wav_lst_valid = pickle.load(fo)

    tgt_file = os.path.join(data_folder, 'remove_oov', 'test-unseen.pkl')
    with open(tgt_file, 'rb') as fo:
        wav_lst_test = pickle.load(fo)
        problem_file = "data/soundspaces_speech/test-unseen/UwV83HsGsw3/5142-33396-0005.pkl"
        if problem_file in wav_lst_test:
            wav_lst_test.remove(problem_file)



    #files = get_all_files(noisy_folder)

    #wav_lst_train = files[: int(len(files) * 0.95)]
    #wav_lst_valid = files[int(len(files) * 0.95) + 1: int(len(files) * 0.975)]
    #wav_lst_test = files[int(len(files) * 0.975) + 1:]

    logger.debug("Creating json files for SoundSpaces...")

    print("Generating train dataset...")
    create_json(wav_lst_train, save_json_train)

    print("Generating valid dataset...")
    create_json(wav_lst_valid, save_json_valid)

    print("Generating test dataset...")
    create_json(wav_lst_test, save_json_test)

    print("Generating full dataset...")
    create_json(wav_lst_train + wav_lst_valid + wav_lst_test, save_json_full)

def skip(*filenames):
    """
    Detects if the SSSpeech data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def remove_punctuation(a_string):
    """Remove all punctuation from string"""
    return a_string.translate(str.maketrans("", "", string.punctuation))


def create_json(wav_lst, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    logger.debug(f"Creating json lists in {json_file}")

    # Processing all the wav files in the list
    json_dict = {}
    SAMPLERATE = 16000
    for wav_file in tqdm.tqdm(wav_lst):  # ex:p203_122.wav

        with open(wav_file, 'rb') as fo:
            data = pickle.load(fo)

        receiver_audio = data['receiver_audio']
        source_audio = data['source_audio']


        source_audio, receiver_audio = process_audio(source_audio, receiver_audio)



        # Example wav_file: p232_001.wav
        noisy_path, filename = os.path.split(wav_file)
        root, noisy_dir = os.path.split(noisy_path)
        noisy_rel_path = os.path.join("{data_root}", os.path.relpath(wav_file, "data/soundspaces_speech"))
        #print(noisy_rel_path)
        #print(receiver_audio.shape)
        # Reading the signal (to retrieve duration in seconds)
        duration = receiver_audio.shape[0] / SAMPLERATE

        # Read text
        snt_id = filename.replace(".pkl", "")
        #print(snt_id)

        json_dict[snt_id] = {
            "data": noisy_rel_path,
            "dereverb": os.path.join("{data_root}", "metric_dereverb", "audio", filename.replace(".pkl", ".wav")),
            "length": duration
        }

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")

def process_audio(source_audio, receiver_audio):
    waveform_length = 40960

    if waveform_length > receiver_audio.shape[0]:
        receiver_audio = np.pad(receiver_audio, (0, max(0, waveform_length - receiver_audio.shape[0])))
    if waveform_length > source_audio.shape[0]:
        source_audio = np.pad(source_audio, (0, waveform_length - source_audio.shape[0]))

    start_index = np.random.randint(0, source_audio.shape[0] - waveform_length) if source_audio.shape[0] != waveform_length else 0

    source_clip = source_audio[start_index: start_index + waveform_length]
    receiver_clip = receiver_audio[start_index: start_index + waveform_length]

    return source_clip, receiver_clip
