# SS-VAM de-biaser recipe

This recipe ports the paper's local SpeechBrain/MetricGAN-U implementation into the SS-VAM release. It includes the acoustic residue metric, static and dynamically updated reverberator variants, dataset preparation, and audio export. It runs with the SpeechBrain version in the root requirements; a separate modified SpeechBrain checkout is not required.

Install the root package and this directory's `extra-requirements.txt`, prepare `data/` as described in the main README, and run commands from the **repository root**. The commands below demonstrate the SoundSpaces pipeline. They are starting configurations, not a claim to reproduce every paper table with the default hyperparameters.

## 1. Bootstrap the reverberators

The recipe evaluates acoustic residue during validation, including for SRMR training, so it requires both visual and audio-only checkpoints and the upstream `data/pretrained-models/rt60_estimator.pth`.

If you do not have existing checkpoints, train initial reverberators using the original SoundSpaces source audio:

```bash
python trainer.py --model wavenet --dataset soundspaces \
  --decode-wav --input-mono --output-mono --use-rgb --remove-oov \
  --model-dir data/models/bootstrap_vis --version v1 --batch-size 16
python trainer.py --model wavenet --dataset soundspaces \
  --decode-wav --input-mono --output-mono --remove-oov \
  --model-dir data/models/bootstrap_no_vis --version v1 --batch-size 16
```

This bootstrap uses the synthetic data's original source audio. The subsequent stages learn from generated pseudo-dry audio; use your intended experimental initialization and settings when reproducing the paper.

## 2. Train the SRMR de-reverberator and export audio

```bash
python recipes/metricgan/train.py recipes/metricgan/hparams/train_dereverb.yaml \
  --device cuda:0 --data_folder data/soundspaces_speech \
  --output_folder data/results/srmr \
  --target_metric srmr --use_devb False \
  --visual_reverberator_checkpoint data/models/bootstrap_vis/v1/best_val.ckpt \
  --blind_reverberator_checkpoint data/models/bootstrap_no_vis/v1/best_val.ckpt
```

The SoundSpaces preparation helper reads the split lists in `remove_oov/` and creates `train_revb.json`, `valid_revb.json`, `test_revb.json`, and `full_revb.json` in the data folder. The full manifest combines training, validation, and unseen test examples; a seen-test export requires a manifest containing that split. Existing complete manifests are reused. Use `--skip_prep True` if providing your own manifests.

Export using the same configuration and output folder, adding `--generate_dataset`:

```bash
python recipes/metricgan/train.py recipes/metricgan/hparams/train_dereverb.yaml \
  --device cuda:0 --data_folder data/soundspaces_speech \
  --output_folder data/results/srmr --target_metric srmr --use_devb False \
  --visual_reverberator_checkpoint data/models/bootstrap_vis/v1/best_val.ckpt \
  --blind_reverberator_checkpoint data/models/bootstrap_no_vis/v1/best_val.ckpt \
  --generate_dataset
```

This restores the best generator from `data/results/srmr/save/` and writes 16 kHz WAVs to `data/soundspaces_speech/metric_dereverb/audio/`. WAV basenames correspond to source pickle basenames; preserve the original dataset's naming convention. Export preserves the directory and overwrites files with matching IDs.

## 3. Train reverberators on the generated audio

Run the visual training command in the root README with `--use-metricgan-dereverb` and `--model-dir data/models/ss_srmr_vis`. Repeat without `--use-rgb` using `data/models/ss_srmr_no_vis`.

These two checkpoints supply the acoustic residue metric for the next stage. Their architecture flags are recovered from each checkpoint's saved `hparams`.

## 4. Train the de-biaser

```bash
python recipes/metricgan/train.py recipes/metricgan/hparams/train_dereverb.yaml \
  --device cuda:0 --data_folder data/soundspaces_speech \
  --output_folder data/results/debias_static \
  --target_metric rt60_bias --use_devb True --use_dynamic_rt60 False \
  --visual_reverberator_checkpoint data/models/ss_srmr_vis/v1/best_val.ckpt \
  --blind_reverberator_checkpoint data/models/ss_srmr_no_vis/v1/best_val.ckpt
```

`--use_devb True` feeds stage-1 `metric_dereverb` outputs into the de-biaser. `--target_metric rt60_bias` selects the acoustic residue objective. `--use_dynamic_rt60 True` enables the original periodic reverberator updates; use a separate output directory for that experiment. Other controls, including `bias_variant`, `update_freq`, and loss weights, are documented in the YAML.

## 5. Export de-biased audio and evaluate

Repeat the stage-4 command with `--generate_dataset`. By default, the acoustic-residue variant exports the held-out test manifest; add `--export_all True` to export the full manifest for reverberator training:

```bash
python recipes/metricgan/train.py recipes/metricgan/hparams/train_dereverb.yaml \
  --device cuda:0 --data_folder data/soundspaces_speech \
  --output_folder data/results/debias_static \
  --target_metric rt60_bias --use_devb True --use_dynamic_rt60 False \
  --visual_reverberator_checkpoint data/models/ss_srmr_vis/v1/best_val.ckpt \
  --blind_reverberator_checkpoint data/models/ss_srmr_no_vis/v1/best_val.ckpt \
  --generate_dataset --export_all True
```

Static outputs go to `metric_debias/rt60/audio/`; dynamic outputs go to `metric_debias/dynamic_rt60/audio/`. The reverberator training loader expects `metric_debias/audio/`. For a new static experiment, create a link after export:

```bash
# Run only when this destination does not already exist.
ln -s rt60/audio data/soundspaces_speech/metric_debias/audio
```

For dynamic training, link `dynamic_rt60/audio` instead. Train with `--use-metricgan-debias`; evaluate a static export with `--version v1 --static-rt60-eval --use-metricgan-debias`, supplying your trained reverberator checkpoint. Keep audio outputs and evaluation flags consistent.

## Acoustic AVSpeech and optional experiments

Use `--data_folder data/acoustic_avspeech` and checkpoints trained with `--dataset avspeech`. The preparation helper creates manifests from numeric WAV/image IDs. Existing curated manifests are preferred when reproducing a specific experiment. De-biased inputs must be generated before stages that consume them.

The optional AVSpeech subset mode expects a supplied `full_sub_revb.json`; the normal mode uses `full_revb.json`. Optional LibriSpeech export requires a supplied `data/soundspaces_speech/ls_revb.json` manifest and the `--generate_libri_dataset --generate_dataset` flags. These optional manifests are not bundled.

DNSMOS is an optional remote scoring path inherited from MetricGAN-U. SRMR and acoustic residue run locally and need no service key. If choosing DNSMOS, provide your own service credentials through `DNSMOS_AUTH_KEY`; no credentials are stored in the repository.

## Checkpoints and licenses

SpeechBrain saves the generator, discriminator, and epoch state below `<output_folder>/save/`. Keep the same output folder and training settings when resuming or exporting. Full GPU training has not been repeated as part of this repository cleanup.

This recipe derives from SpeechBrain's MetricGAN-U implementation (Szu-Wei Fu and contributors), under Apache License 2.0. See the root third-party notices and license copy.
