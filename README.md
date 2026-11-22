# Indian Sign Language Word Recognition from Video

Recognise isolated ISL signs (one word per clip) from short videos, to support accessibility tools.

**Research question:** for *low-resource* sign language recognition, which input works better -
**raw RGB frames** or **pose keypoints** - in terms of accuracy *and* speed?

| | Pipeline A - raw frames | Pipeline B - pose keypoints |
|---|---|---|
| Input | 16 RGB frames, 224×224 | 32 frames of MediaPipe Holistic landmarks (body + both hands) |
| Spatial features | Pretrained CNN per frame (MobileNetV2 / ResNet18 / EfficientNet-B0) | Normalised coordinates, wrist-relative hand shape, velocity |
| Temporal model | (Bi)LSTM - LRCN (Donahue et al., 2015) | (Bi)LSTM with attention pooling **or** Transformer encoder |
| Model size | ~5-13 M params, 9.7-12.4 GFLOPs/clip (MobileNetV2 / EfficientNet-B0, measured) | 0.5-0.8 M params, 0.03-0.05 GFLOPs/clip (measured) |

Dataset: **INCLUDE / INCLUDE-50** (Sridhar et al., 2020). See [`data/README.md`](data/README.md).

## Results (test set, 137 videos)

Same split and test set for every model, one run per model. Speed was measured on one laptop (RTX 2050);
the last column is the mean of 8 runs on one real video.

| Model | Params | GFLOPs / clip | Test top-1 | CPU, 1 thread (model only) | Raw video to prediction (CPU) |
|---|---|---|---|---|---|
| Pose-Transformer (B) | 0.48 M | 0.032 | 0.934 (128/137) | 1.0 ms | 1858 ms |
| Pose-LSTM (B) | 0.76 M | 0.047 | 0.883 (121/137) | 1.5 ms | 1867 ms |
| CNN-LSTM MobileNetV2 (A) | 5.40 M | 9.69 | 0.927 (127/137) | 288 ms | 746 ms |
| CNN-LSTM EfficientNet-B0 (A) | 7.18 M | 12.41 | 0.978 (134/137) | 397 ms | 704 ms |

* **Accuracy:** raw frames and keypoints are not distinguishable on 137 test videos (MobileNetV2 0.927 and
  Pose-Transformer 0.934 differ by one video). EfficientNet-B0 is a single run: an earlier run of the same
  configuration reached 0.934, so a few videos of difference between runs are normal.
* **Size:** the pose models have 0.5 to 0.8 M parameters against 5.4 to 7.2 M, and need 200 to 400 times fewer FLOPs.
* **Speed:** as models the pose networks take 1 to 1.5 ms against 288 to 397 ms, but from a raw video the frames
  pipeline is about 2.5 times faster, because MediaPipe alone takes about 1.3 s per video.
* **Caveats:** one seed per model; the random split probably puts near-duplicate takes of one session in both
  train and test, so accuracies are likely higher than for new signers; keypoints come from cached 224 x 224 frames
  (hands are found in only about 55 to 67 % of the frames); timings vary by 20 to 40 % between sessions.

Full report: [`reports/Team-29_Report.pdf`](reports/Team-29_Report.pdf).

---

## Quick start (3 minutes, CPU, synthetic data)

Python 3.12 is recommended. MediaPipe 0.10.21, which is only needed to extract keypoints, does not install on Python 3.13.

```bash
git clone https://github.com/Rani553/indian-sign-language-video-recognition.git && cd indian-sign-language-video-recognition
pip install -r requirements.txt
python scripts/make_dummy_data.py --data-root /tmp/isl_dummy --fake-keypoints
pytest                                   # data, features, models, evaluation, full train->eval loop
```

## Full pipeline (real data)

Colab/Kaggle notebooks for every step are in [`notebooks/colab/`](notebooks/colab).

```bash
# 1. Data (Member 1) - CPU session
python -m src.data.download --classes configs/include50_classes.txt
python -m src.data.split
python -m src.features.extract_frames

# 2. Keypoints (Member 3) - CPU session, needs MediaPipe 0.10.21
pip install -r requirements-pose.txt
python -m src.features.extract_keypoints

# 3. Training - GPU session (resumes automatically after a disconnect)
python -m src.training.train_cnn_lstm --config configs/cnn_lstm.yaml          # Member 2
python -m src.training.train_pose --config configs/pose_lstm.yaml             # Member 3
python -m src.training.train_pose --config configs/pose_transformer.yaml      # Member 3

# 4. Evaluation and comparison (Member 4)
python -m src.evaluation.evaluate --checkpoint results/*/best.pt
python -m src.evaluation.benchmark_speed --checkpoint results/*/best.pt --device cpu cuda --video <test video>
python -m src.evaluation.compare
streamlit run app/demo.py
```

Every script accepts `--tiny` (5 classes, few videos, 2 epochs) for a quick check and
`--set key=value` to override any config value, e.g.
`--set model.backbone=resnet18 train.epochs=20 experiment=cnn_lstm_resnet18`.

## Colab / Kaggle notes
* Paths are detected automatically: Colab uses Google Drive (`MyDrive/isl_data`, `MyDrive/isl_results`)
  so nothing is lost on disconnect; Kaggle uses `/kaggle/working`. Override with `ISL_DATA_ROOT` /
  `ISL_RESULTS_ROOT` or `--set paths.data_root=...`.
* Videos are decoded **once** into small caches (frames `.npy`, keypoints `.npz`); training never
  touches the raw videos. Share `processed/` as a Kaggle Dataset or a shared Drive folder.
* `last.pt` is written every epoch; re-running the same command **resumes** training.
* Mixed precision and gradient accumulation keep memory low on free T4/P100 GPUs.
* **MediaPipe:** versions ≥ 0.10.30 removed the `solutions.holistic` API. Use `mediapipe==0.10.21`
  (installs numpy < 2 → restart the runtime once). Run extraction in a **CPU** session.

## Project structure
```
configs/            YAML configs (model configs inherit data.yaml)
src/data/           download, split, datasets, augmentation, EDA
src/features/       frame extraction, MediaPipe extraction, keypoint feature engineering
src/models/         cnn_lstm.py, pose_lstm.py, pose_transformer.py, factory
src/training/       generic Trainer + training entry points
src/evaluation/     metrics, evaluation, speed benchmark, comparison
src/inference/      Predictor: raw video -> top-k signs (used by demo and benchmark)
app/demo.py         Streamlit demo
notebooks/colab/    one ready-to-run notebook per team member + quickstart
scripts/            setup (Colab / Kaggle), dummy data, ablation runners
tests/              pytest suite (runs on CPU with synthetic data; used by CI)
reports/            final report (PDF) and the ablation tables
```

## Outputs
```
<results_root>/<experiment>/
    best.pt, last.pt, config.yaml, history.csv, curves.png, train.log
    eval_test/  metrics.json, per_class.csv, predictions.csv, confusion_matrix.png, top_confusions.json
    benchmark_cpu.json, benchmark_cuda.json
<results_root>/comparison/  comparison.csv/.md, accuracy_vs_latency.png, accuracy_bars.png
```

## Team
Four members, one module each. Each member's tasks were tracked as a GitHub issue and developed on
its own `feature/*` branch, merged into `develop` through pull requests.

| Member | Part |
|---|---|
| M1 Rani Kumari | Data pipeline, configs, split, trainer |
| M2 Nirmal Mewada | Pipeline A: CNN + LSTM |
| M3 Pragati Chauhan | Pipeline B: keypoints, Pose-LSTM, Pose-Transformer |
| M4 Rohit Malviya | Evaluation, speed benchmark, comparison, demo, tests and CI, report |

## References
1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign
   Language Recognition.* ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and
   Description.* CVPR 2015. https://arxiv.org/abs/1411.4389
3. C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* 2019.
   https://arxiv.org/abs/1906.08172
