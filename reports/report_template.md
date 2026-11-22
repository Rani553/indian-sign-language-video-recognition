# Indian Sign Language word recognition from video: raw frames or pose keypoints?

**Course project, team of four.** Data pipeline (M1), Pipeline A raw frames (M2), Pipeline B pose keypoints (M3),
evaluation, benchmark, demo and this report (M4).

> All numbers come from `src/evaluation/evaluate.py`, `benchmark_speed.py` and `compare.py` (outputs in
> `results/comparison/`), run on one laptop for every model. Sections 3 and 4 summarise `reports/sections/pipeline_a.md`
> and `pipeline_b.md`.

## Abstract

We ask whether an Indian Sign Language word is better recognised from raw video frames or from pose keypoints, in
accuracy and in speed, on a small dataset (INCLUDE-50: 50 signs, 958 videos, our own 684 / 137 / 137 split).
Pipeline A feeds 16 RGB frames through a pretrained CNN and an LSTM; Pipeline B extracts body and hand landmarks with
MediaPipe and feeds them to a small LSTM or Transformer. On the 137 test videos the best frames model and the best pose
model are within the noise of each other (MobileNetV2 CNN-LSTM 0.927 and Pose-Transformer 0.934; a second EfficientNet-B0
checkpoint reaches 0.978, see Section 6.1 for why this number is treated with care). The models differ strongly in cost:
the pose models need 200 to 400 times fewer FLOPs and run in 1 to 5 ms, but from a raw video file the frames pipeline is
about 2.5 times faster (0.7 s against 1.9 s on CPU), because MediaPipe alone takes 1.3 s per video. The main caveat is that
the random split probably puts near-duplicate takes of one session in both train and test, so all accuracies are likely
higher than for new signers; one seed per model and 137 test videos also make differences of a few videos unreliable.

## 1. Problem and research question

Isolated sign recognition: one Indian Sign Language word per short video, for accessibility tools.
**Question:** for a *low-resource* sign language dataset, which input is better, **raw RGB frames** or
**pose keypoints**, in terms of accuracy *and* speed?

## 2. Data

* **INCLUDE-50** (Sridhar et al., 2020): 50 signs, 958 videos, 14 to 25 videos per class (mean 19.2),
  15 categories (adjectives are the largest, 233 videos). Official class list: `configs/include50_classes.txt`.
* **Split:** our own stratified split, seed 42, per video: 684 train, 137 validation, 137 test. Every class is in every
  split. It is **not** the official INCLUDE-50 split, so our numbers cannot be compared with published ones.
* **Keypoint quality:** MediaPipe found the body in 99.1 % of frames, the left hand in 66.6 % and the right hand in 54.8 %
  (`results/eda/keypoint_detection.png`).
* **Near-duplicate takes (important for how to read every accuracy).** A signer records a sign several times in a row
  (consecutive `MVI_<number>` files). Because the split is random per video, takes of one session land in train and in
  test: **127 of 137 test videos (93 %)** have a same-class training video within 2 file numbers, and 203 of 283
  runs of consecutive takes are spread over more than one split (`src/data/eda.py`, `split_leak_check.json`).
  This is a heuristic (there is no signer id in the data). It means the test accuracy is probably higher than it would be
  for new recording sessions or new signers. See Section 7.

## 3. Pipeline A: raw frames (CNN + LSTM)

Summary of `reports/sections/pipeline_a.md`. Each clip becomes 16 RGB frames of 224 x 224; a pretrained CNN (MobileNetV2,
ResNet18 or EfficientNet-B0) turns every frame into a vector; a 1-layer bidirectional LSTM (hidden size 256) reads the
vectors and its outputs are mean-pooled and classified. The CNN is frozen for 3 epochs and then fine-tuned (AdamW,
label smoothing 0.1, augmentation, early stopping, one seed). Key ablation findings (test videos correct, of 137): a frozen
CNN does not work (62), last-step pooling is much worse than mean or max pooling (102 against 127 and 128), and the
rest is within noise: EfficientNet-B0 128 and MobileNetV2 127, 8 frames 125 (half the training time), fine-tuning from the
first epoch 126, ResNet18 118 (probably not converged). MobileNetV2 is the baseline used here because it is the lightest.

## 4. Pipeline B: pose keypoints

Summary of `reports/sections/pipeline_b.md`. MediaPipe Holistic finds the body (33 points) and both hands (21 points each)
in every frame; gaps are interpolated, and each of 32 frames becomes 536 features (upper-body pose, hand coordinates,
wrist-relative hand shape, velocity, hand-presence flags). Two sequence models read them: a Pose-LSTM with attention
pooling (0.76 M parameters) and a Pose-Transformer with a CLS token (0.48 M). Key ablation findings (validation, because the
test set was kept for the two baselines): none of 14 runs is distinguishable from the baseline (126 to 131 videos of 137).
Hands alone or body alone are about as good as both, 16 frames are as good as 32, and removing velocity or augmentation
changes nothing measurable.

## 5. Evaluation protocol

* **Same test set for every model:** the 137 test videos, no augmentation, fp32, fixed frame sampling
  (`src/evaluation/evaluate.py`). Checkpoints were chosen on the validation set only.
* **Metrics:** top-1 and top-5 accuracy, macro F1, a 95 % Wilson interval for top-1, per-class table, confusion matrix.
  With 137 videos one video is 0.73 points and the interval is about +-4 points, so differences of a few videos are noise.
  Pairs of models are compared with a paired exact test on the videos where exactly one model is right
  (`notebooks/02_results_analysis.ipynb`).
* **Speed (`src/evaluation/benchmark_speed.py`), all models on the same machine:**
  parameters and checkpoint size; GFLOPs of one clip (2 FLOPs per multiply-add; LSTMs are added by hand because
  PyTorch's counter skips them); latency of the model alone (batch 1, 10 warm-up runs, 50 timed runs) on CPU with one
  thread (stable) and on GPU; throughput with batches of 16; and **end-to-end time from a raw video file**, split into
  decode and resize, features (CNN input, or MediaPipe plus feature engineering) and model.
* **Hardware:** one laptop for every model: Intel 12th-gen Core CPU (Family 6 Model 154, 12 threads), NVIDIA GeForce RTX 2050
  (CUDA 12.4), Windows 11, on mains power, Windows "Balanced" power plan, nothing else running. Model-only numbers:
  Python 3.13, PyTorch 2.6.0+cu124; all four checkpoints were timed in one session, back to back (100 timed runs after 20
  warm-up runs). End-to-end numbers: a separate Python 3.12.5 environment with PyTorch 2.15.0 (CPU) and
  MediaPipe 0.10.21, because MediaPipe does not install on Python 3.13; all three models were timed there, so the
  end-to-end comparison is like for like. Each end-to-end number is the mean of 8 runs on one video (`MVI_4992.MOV`) after one warm-up.
* The predictor uses exactly the preprocessing of training; a test checks that its input equals the dataset's input
  for the same cached frames (`tests/test_pipeline.py`).

## 6. Results

### 6.1 Accuracy and speed

| Model                    | Pipeline                   |   Params (M) |   GFLOPs |   Top-1 | 95% CI      |   Top-5 |   Macro-F1 |   CPU ms (1 thread) |   GPU ms |   End-to-end CPU ms |
|--------------------------|----------------------------|--------------|----------|---------|-------------|---------|------------|---------------------|----------|---------------------|
| cnn_lstm_efficientnet_b0 | Pipeline A: raw frames     |        7.183 |   12.406 |   0.978 | 0.938-0.993 |   1.000 |      0.976 |             397.002 |   38.899 |             703.999 |
| pose_transformer         | Pipeline B: pose keypoints |        0.482 |    0.032 |   0.934 | 0.880-0.965 |   0.985 |      0.925 |               1.011 |    1.098 |            1857.565 |
| cnn_lstm_mobilenet_v2    | Pipeline A: raw frames     |        5.399 |    9.685 |   0.927 | 0.871-0.960 |   1.000 |      0.928 |             288.153 |   31.965 |             746.053 |
| pose_lstm                | Pipeline B: pose keypoints |        0.759 |    0.047 |   0.883 | 0.819-0.927 |   0.978 |      0.879 |               1.507 |    0.860 |            1866.943 |

Test set, 137 videos, evaluated by `src/evaluation/evaluate.py` for every model:

| Model | Top-1 | 95 % interval | Top-5 | Macro F1 | Params |
|---|---|---|---|---|---|
| Pose-Transformer | 0.934 (128/137) | 0.880 to 0.965 | 0.985 | 0.925 | 0.48 M |
| Pose-LSTM | 0.883 (121/137) | 0.819 to 0.927 | 0.978 | 0.879 | 0.76 M |
| CNN-LSTM (MobileNetV2) | 0.927 (127/137) | 0.871 to 0.960 | 1.000 | 0.928 | 5.40 M |
| CNN-LSTM (EfficientNet-B0), checkpoint we received | 0.978 (134/137) | 0.938 to 0.993 | 1.000 | 0.976 | 7.18 M |

![accuracy against speed](../results/comparison/accuracy_vs_latency.png)
![accuracy](../results/comparison/accuracy_bars.png)

Speed, model only, one clip, median of 100 runs after 20 warm-up runs (same laptop, same session for all models):

| Model | GFLOPs | Checkpoint | CPU, 1 thread | CPU, 12 threads | GPU (RTX 2050) | Throughput GPU, batch 16 |
|---|---|---|---|---|---|---|
| Pose-Transformer | 0.032 | 1.9 MB | 1.0 ms | 1.7 ms | 1.1 ms | 14,004 clips/s |
| Pose-LSTM | 0.047 | 3.0 MB | 1.5 ms | 5.5 ms | 0.9 ms | 15,507 clips/s |
| CNN-LSTM (MobileNetV2) | 9.69 | 21.8 MB | 288 ms | 215 ms | 32 ms | 29 clips/s |
| CNN-LSTM (EfficientNet-B0) | 12.41 | 29.0 MB | 397 ms | 264 ms | 39 ms | 25 clips/s |

The pose models need 200 to 400 times fewer FLOPs than the CNN-LSTMs. The tiny pose models run slower with 12 CPU threads
than with one (thread start-up costs more than the work), so the one-thread number is the one to compare models with.
Timing noise on this laptop is real: the same model measured on different occasions differed by about 20 to 40 %
(for example MobileNetV2 on one thread: 333 ms in an earlier session, 288 ms in the final one), so only differences of
a factor of two or more should be read as real.

**How to read the EfficientNet-B0 row.** The checkpoint we received reaches 0.978 on the test set (134 of 137), but the
Pipeline A report gives 0.934 (128 of 137) for its EfficientNet-B0 run, with a different best validation accuracy (0.971 at
epoch 20 in our file, 0.964 at epoch 19 in the report). So it is a different run of the same configuration. A difference of six
videos is possible between two runs, but we could not confirm how this run was trained, so we do **not** use it to claim that
EfficientNet-B0 is the best model. The MobileNetV2 checkpoint reproduces its reported 0.927 exactly.

### 6.2 Where the time goes (end to end)

A user of the demo pays for the whole chain from a raw video file, not for the model alone. One real INCLUDE video
(`MVI_4992.MOV`, class "summer"), CPU, mean of 8 runs after one warm-up, all four models in one session and one environment:

| Model | Decode and resize | Features | Model | **Total** |
|---|---|---|---|---|
| CNN-LSTM (EfficientNet-B0) | 512 ms | 7 ms | 185 ms | **704 ms** |
| CNN-LSTM (MobileNetV2) | 567 ms | 8 ms | 172 ms | **746 ms** |
| Pose-Transformer | 520 ms | 1335 ms (MediaPipe) | 2 ms | **1858 ms** |
| Pose-LSTM | 523 ms | 1341 ms (MediaPipe) | 3 ms | **1867 ms** |

Although the pose models are 200 to 400 times cheaper as models, the frames pipeline is about **2.5 times faster end to
end** on CPU, because MediaPipe takes about 1.3 s per video, hundreds of times longer than the model. The pose report
estimated 2 to 5 s per video; this machine is at the fast end of that range. Decoding the video (about 0.5 s) is the
largest cost of the frames pipeline, because the loader reads the whole video before it keeps 24 frames. The two frames
models are not distinguishable here (704 against 746 ms; in other runs the order changed). Caveats: one video, one machine,
and run-to-run differences of 20 to 40 % (the EfficientNet total was 1108 and 1417 ms in two runs made while the laptop was
busy); the ranking of the two pipelines did not change in any run.

### 6.3 Errors

Misclassified test videos (of 137): EfficientNet-B0 3, Pose-Transformer 9, MobileNetV2 10, Pose-LSTM 16
(`results/<model>/eval_test/top_confusions.json`, `per_class.csv`).

* **Shared by both pipelines:** *boy* and *girl* (both MobileNetV2 and Pose-Transformer confuse them twice), *paint*
  is wrong in all four models, *red* in two, and *short* / *smalllittle* in three of the four.
* **Frames only:** *short* and *smalllittle* are swapped in both directions by both CNN-LSTMs; they are the most
  frequent frames errors. The pose models make these errors rarely (Pose-Transformer none, Pose-LSTM two).
* **Pose only:** *time* / *teacher*, *monday* / *hot* and *election* / *shoes* appear in the pose models; the frames
  models get them right. These pairs differ mainly in hand shape, which MediaPipe found in only 55 to 67 % of frames.
* The mistakes are therefore partly different, which is the only sign that the two inputs carry different information.
  We did not test whether combining them would help.

## 7. Discussion

* **Accuracy:** with the two checkpoints that reproduce their reports, there is no evidence of a difference between the
  pipelines: MobileNetV2 CNN-LSTM 0.927 and Pose-Transformer 0.934 (128 against 127 videos) are inside each other's
  95 % interval; the paired exact test on the videos where exactly one is right gives 5 against 6, p = 1.0. The Pose-LSTM
  (0.883) is 6 to 12 videos behind the others; against the MobileNetV2 CNN-LSTM p = 0.24 (12 against 6) and against the
  Pose-Transformer p = 0.09 (10 against 3), so even this gap is not conclusive. The EfficientNet-B0 checkpoint (0.978)
  is 7 videos ahead of MobileNetV2 with none behind (p = 0.016) and 14 ahead of the Pose-LSTM (p = 0.001), but it is the
  checkpoint whose result differs from its report (Section 6.1), and with one run per model and a split that may share
  sessions between train and test we do not draw a conclusion from it. The honest summary: the frames and keypoint
  pipelines reach similar accuracy on this dataset.
* **Speed:** the answer depends on what is measured. As a model on cached input, the pose models are far cheaper
  (0.03 to 0.05 against 10 to 12 GFLOPs, 1 to 2 ms against 290 to 400 ms on one CPU thread, 2 to 3 MB against 22 to 29 MB).
  From a raw video file, the frames pipeline is faster (about 0.7 s against 1.9 s on CPU), because MediaPipe costs about
  1.3 s per video. For a demo or an offline tool that starts from video files, the frames pipeline is the simpler and faster
  choice here. The pose models would pay off where keypoints are computed anyway or can be computed once and reused
  (many models on the same keypoints, a GPU or phone pose model, streaming where keypoints are extracted frame by frame),
  and where a small model file matters. We did not measure a GPU MediaPipe, a phone, or a webcam stream.
* **Reading the numbers (limitations):**
  * *Near-duplicate takes (suspected, not proven).* We assume that consecutive file numbers of one class are takes of
    the same session; the dataset has no signer id to confirm it. With a gap of one file number, test videos that have a
    training take were classified better than the others: Pose-Transformer 0.954 (109 videos) against 0.857 (28
    videos), Pose-LSTM 0.927 against 0.714. With a gap of two, only 10 videos have no training take and the
    picture is mixed (Transformer 0.937 against 0.900, LSTM 0.898 against 0.700). The groups are small, so this is
    weak evidence. It is consistent with a random split being easier than recognising a new session, but it does not
    show how much. A split by recording session would answer it (all models would have to be retrained).
  * One seed per configuration, 137 test videos: differences of a few videos are not reliable.
  * Own split, not the official INCLUDE-50 split.
  * Keypoints come from cached 224 x 224 frames, not from the original videos; hands were found in only 55 to 67 % of frames.
  * All signers are from one recording setting; other signers, backgrounds and regions are untested.
  * Speed was measured on one laptop (RTX 2050, Balanced power plan); other hardware changes absolute times, and tiny
    models even run slower with more CPU threads. End-to-end time was measured on one video only.

## 8. Demo and reproducibility

`streamlit run app/demo.py` (upload a video, compare models, see the time per stage).
`pytest` runs the whole chain on synthetic data (CI on every pull request, `.github/workflows/ci.yml`).
Exact commands: README, section "Full pipeline".

## 9. Conclusion

On this dataset and split, raw frames and pose keypoints reach similar test accuracy (MobileNetV2 CNN-LSTM 0.927,
Pose-Transformer 0.934, Pose-LSTM 0.883); the differences between the first two are inside the noise of 137 test videos.
The pose models are far smaller and cheaper as models (about 300 times fewer FLOPs), but from a raw video the whole
pipeline is about 2.5 times slower than the frames pipeline on CPU, because extracting keypoints with MediaPipe costs
more than the frames models themselves. So neither input is better on both axes: frames are simpler and faster end to
end here, keypoints give a much smaller model. These conclusions hold with the caveats of Section 7: one seed per model,
a random split in which near-duplicate takes of one session probably sit in both train and test, one recording setting,
and speed measured on one laptop and one video.

## 10. Team contributions

| Member | Part |
|---|---|
| M1 Rani Kumari | Data pipeline, configs, split, trainer |
| M2 Nirmal Mewada | Pipeline A: CNN + LSTM |
| M3 Pragati Chauhan | Pipeline B: keypoints, Pose-LSTM, Pose-Transformer |
| M4 Rohit Malviya | Evaluation, speed benchmark, comparison, demo, tests and CI, this report |

## References

1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.*
   ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and Description.* CVPR 2015.
   https://arxiv.org/abs/1411.4389
3. C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* 2019. https://arxiv.org/abs/1906.08172
4. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
5. E. B. Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 1927 (confidence interval for a proportion).
