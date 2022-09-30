from concurrent.futures import ProcessPoolExecutor
from glob import glob
import os

from tqdm import tqdm
import librosa
from scipy.io import wavfile


def run(batch_files):
    for file in tqdm(batch_files):
        new_file = file.replace('LibriSpeech', 'LibriSpeech-wav')
        new_dir = os.path.dirname(new_file)
        os.makedirs(new_dir, exist_ok=True)
        wav, sr = librosa.load(file, sr=None)
        wavfile.write(new_file, sr, wav)


files = glob('data/sounds/speech/LibriSpeech/dev-clean/**/*.flac', recursive=True)
num_process = 20
step = len(files) // num_process + 1
with ProcessPoolExecutor(max_workers=num_process) as executor:
    for i in range(num_process):
        executor.submit(run, files[i * step: (i+1) * step])

