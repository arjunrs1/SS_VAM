# audio_synthesis

## Non-visual diffusion model:

Train:
python diff_trainer.py --model-dir data/models/<MODEL_NAME> --use-pretrained --use-frozen-dereverb

Test:
python diff_trainer.py --model-dir data/models/<MODEL_NAME> --use-pretrained --use-frozen-dereverb --test

## Visual diffusion model:

Train:
python diff_trainer.py --model-dir data/models/<MODEL_NAME> --use-pretrained --use-frozen-dereverb --use-rt60-feature

Test:
python diff_trainer.py --model-dir data/models/<MODEL_NAME> --use-pretrained --use-frozen-dereverb --use-rt60-feature --test
