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
from novas.datasets.avspeech_dataset import AVSpeechDataset
dataset = AVSpeechDataset

logger = logging.getLogger(__name__)
LEXICON_URL = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
TRAIN_JSON = "train_revb.json"
TRAIN_JSON2 = "train_revb2.json"
TRAIN_JSON3 = "train_revb3.json"
TRAIN_JSON4 = "train_revb4.json"
TRAIN_JSON5 = "train_revb5.json"

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
    >>> data_folder = '/path/to/datasets/AVSpeech'
    >>> save_folder = 'exp/AVSpeech_exp'
    >>> prepare_data(data_folder, save_folder)
    """

    if skip_prep:
        return

    # Setting ouput files
    save_json_train = os.path.join(save_folder, TRAIN_JSON)
    save_json_train2 = os.path.join(save_folder, TRAIN_JSON2)
    save_json_train3 = os.path.join(save_folder, TRAIN_JSON3)
    save_json_train4 = os.path.join(save_folder, TRAIN_JSON4)
    save_json_train5 = os.path.join(save_folder, TRAIN_JSON5)

    save_json_valid = os.path.join(save_folder, VALID_JSON)
    save_json_test = os.path.join(save_folder, TEST_JSON)
    save_json_full = os.path.join(save_folder, FULL_JSON)


    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_test, save_json_valid, save_json_full):
        logger.info("Preparation completed in previous run, skipping.")
        return


    noisy_folder = os.path.join(data_folder, "audio")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    logger.debug("Creating lexicon...")
    logger.info("Creating json files for AVSpeech...")

    logger.debug("Collecting files...")
    extension = [".wav"]

    files = get_all_files(noisy_folder)

    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    wav_lst_train = files[: int(len(files) * 0.2)]
    wav_lst_train2 = files[int(len(files) * 0.2) + 1: int(len(files) * 0.4)]
    wav_lst_train3 = files[int(len(files) * 0.4) + 1: int(len(files) * 0.6)]
    wav_lst_train4 = files[int(len(files) * 0.6) + 1: int(len(files) * 0.8)]
    wav_lst_train5 = files[int(len(files) * 0.8) + 1: int(len(files) * 0.95)]

    wav_lst_valid = files[int(len(files) * 0.95) + 1: int(len(files) * 0.975)]
    wav_lst_test = files[int(len(files) * 0.975) + 1:]

    logger.debug("Creating json files for AVSpeech...")

    print("Generating train dataset split 1...")
    create_json(wav_lst_train, save_json_train)

    print("Generating train dataset split 2...")
    create_json(wav_lst_train2, save_json_train2)

    print("Generating train dataset split 3...")
    create_json(wav_lst_train3, save_json_train3)

    print("Generating train dataset split 4...")
    create_json(wav_lst_train4, save_json_train4)

    print("Generating train dataset split 5...")
    create_json(wav_lst_train5, save_json_train5)

    print("Generating valid dataset...")
    create_json(wav_lst_valid, save_json_valid)

    print("Generating test dataset...")
    create_json(wav_lst_test, save_json_test)

    print("Generating full dataset...")
    create_json(files, save_json_full)

def skip(*filenames):
    """
    Detects if the AVSpeech data_preparation has been already done.
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
    for wav_file in tqdm.tqdm(wav_lst):  # ex:p203_122.wav

        try:
            # Example wav_file: p232_001.wav
            noisy_path, filename = os.path.split(wav_file)
            root, noisy_dir = os.path.split(noisy_path)
            devb_wav_file = os.path.join(root, "metric_dereverb/audio", filename)
            noisy_rel_path = os.path.join("{data_root}", noisy_dir, filename)
            devb_rel_path = os.path.join("{data_root}", "metric_dereverb/audio", filename)
            img_path = os.path.join("{data_root}", "img", filename.replace(".wav", ".png"))


            # Reading the signal (to retrieve duration in seconds)
            signal = read_audio(wav_file)
            devb_signal = signal
            duration = signal.shape[0] / SAMPLERATE
            devb_duration = devb_signal.shape[0] / SAMPLERATE

            # Read text
            snt_id = filename.replace(".wav", "")

            json_dict[snt_id] = {
                "noisy_wav": noisy_rel_path,
                "length": duration,
                "devb_wav": devb_rel_path,
                "devb_length": devb_duration,
                "src_rgb": img_path
            }
        except:
            print("INVALID FILE:")
            print(wav_file)

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")
