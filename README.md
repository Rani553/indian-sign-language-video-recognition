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

---

## Quick start (3 minutes, CPU, synthetic data)

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
scripts/            setup, dummy data, ablations, git bootstrap, file ownership
tests/              pytest suite (runs on CPU with synthetic data; used by CI)
reports/            report template and per-pipeline sections
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
Four members, one module each. Each member's tasks are tracked as a GitHub issue and developed on
its own `feature/*` branch, merged into `develop` through pull requests.

## References
1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign
   Language Recognition.* ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and
   Description.* CVPR 2015. https://arxiv.org/abs/1411.4389
3. C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* 2019.
   https://arxiv.org/abs/1906.08172
