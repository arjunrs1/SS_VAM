# Third-party notices

This research release builds on several projects. Preserve their copyright notices and comply with the license applying to each component.

- The original repository's Apache License 2.0 is retained in `LICENSE`.
- Visual Acoustic Matching / AViTAr code (including the `novas` models, datasets, and training utilities) derives from Meta's [Visual Acoustic Matching](https://github.com/facebookresearch/visual-acoustic-matching), distributed under CC BY-NC 4.0. See `third_party/licenses/visual-acoustic-matching.txt`; those portions retain that license.
- `recipes/metricgan` derives from SpeechBrain's MetricGAN-U recipe, under Apache License 2.0. See `third_party/licenses/speechbrain.txt`. SS-VAM adds the acoustic residue metric and associated training logic.
- WaveNet/DiffWave-derived files retain their existing source notices. Dependencies are distributed separately under their respective licenses.

The root license does not replace the licenses of third-party portions. Datasets and pretrained weights have their own terms and are not included in this repository.
