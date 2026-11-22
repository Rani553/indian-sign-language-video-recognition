# Indian Sign Language word recognition from video: raw frames or pose keypoints?

**Course project, team of four.** Data pipeline (M1), Pipeline A raw frames (M2), Pipeline B pose keypoints (M3),
evaluation, benchmark, demo and this report (M4).

> How to use this template. Numbers marked **[TBD]** must be filled from `results/comparison/` after
> running `evaluate.py`, `benchmark_speed.py` and `compare.py` on **one machine** for **all** models
> (`notebooks/colab/m4_evaluation_demo.ipynb`). Numbers without a mark were checked by the evaluation code.
> Sections 3 and 4 summarise `reports/sections/pipeline_a.md` and `pipeline_b.md`; do not copy them whole.

## Abstract

[TBD, 5-6 sentences: the question, the two pipelines, the data, the main accuracy and speed result, the main caveat.]

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

[Summarise `reports/sections/pipeline_a.md`: 16 RGB frames of 224 x 224, pretrained CNN per frame (MobileNetV2,
ResNet18, EfficientNet-B0), (Bi)LSTM with mean pooling, CNN frozen for 3 epochs then fine-tuned. Key ablation findings.]

## 4. Pipeline B: pose keypoints

[Summarise `reports/sections/pipeline_b.md`: MediaPipe Holistic, 32 frames x 536 features (pose, wrist-relative hand shape,
velocity), Pose-LSTM with attention pooling (0.76 M parameters) and Pose-Transformer with a CLS token (0.48 M).
Key ablation findings: none distinguishable from the baseline.]

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
  Python 3.13, PyTorch 2.6.0+cu124. End-to-end numbers: a separate Python 3.12.5 environment with PyTorch 2.15.0 (CPU) and
  MediaPipe 0.10.21, because MediaPipe does not install on Python 3.13; all three models were timed there, so the
  end-to-end comparison is like for like. Each end-to-end number is the mean of 8 runs on one video (`MVI_4992.MOV`) after one warm-up.
* The predictor uses exactly the preprocessing of training; a test checks that its input equals the dataset's input
  for the same cached frames (`tests/test_pipeline.py`).

## 6. Results

### 6.1 Accuracy and speed

[TBD: paste `results/comparison/comparison.md`.]

Already reproduced by the evaluation code (test set, 137 videos):

| Model | Top-1 | 95 % interval | Top-5 | Macro F1 | Params |
|---|---|---|---|---|---|
| Pose-Transformer | 0.934 (128/137) | 0.880 to 0.965 | 0.985 | 0.925 | 0.48 M |
| Pose-LSTM | 0.883 (121/137) | 0.819 to 0.927 | 0.978 | 0.879 | 0.76 M |
| CNN-LSTM (MobileNetV2) | [TBD] | [TBD] | [TBD] | [TBD] | 5.40 M |
| CNN-LSTM (EfficientNet-B0) | [TBD] | [TBD] | [TBD] | [TBD] | 7.18 M |

![accuracy against speed](../results/comparison/accuracy_vs_latency.png)
![accuracy](../results/comparison/accuracy_bars.png)

Speed, model only, one clip, median of 100 runs after 20 warm-up runs (same laptop for all models):

| Model | GFLOPs | Checkpoint | CPU, 1 thread | CPU, 12 threads | GPU (RTX 2050) | Throughput GPU, batch 16 |
|---|---|---|---|---|---|---|
| Pose-Transformer | 0.032 | 1.9 MB | 0.85 ms | 2.0 ms | 1.1 ms | 13,320 clips/s |
| Pose-LSTM | 0.047 | 3.0 MB | 1.4 ms | 7.9 ms | 1.0 ms | 12,261 clips/s |
| CNN-LSTM (MobileNetV2) | 9.69 | 21.8 MB | 333 ms | 243 ms | 32 ms | 29 clips/s |

The pose models need about 200 to 300 times fewer FLOPs than the MobileNetV2 CNN-LSTM. The tiny pose models run slower
with 12 CPU threads than with one (thread start-up costs more than the work), so the one-thread number is the one to
compare models with. Three repeated runs of the Pose-Transformer gave 2.3, 2.0 and 2.0 ms with 12 threads.
**[TBD: CNN-LSTM (EfficientNet-B0) row, once its checkpoint is available.]**

### 6.2 Where the time goes (end to end)

A user of the demo pays for the whole chain from a raw video file, not for the model alone. One real INCLUDE video
(`MVI_4992.MOV`, class "summer"), CPU, mean of 8 runs, same environment for all models:

| Model | Decode and resize | Features | Model | **Total** |
|---|---|---|---|---|
| CNN-LSTM (MobileNetV2) | 532 ms | 8 ms | 155 ms | **695 ms** |
| Pose-Transformer | 500 ms | 1292 ms (MediaPipe) | 2 ms | **1795 ms** |
| Pose-LSTM | 501 ms | 1272 ms (MediaPipe) | 3 ms | **1776 ms** |

Although the pose models are about 100 to 300 times cheaper as models, the frames pipeline is about **2.6 times faster
end to end** on CPU, because MediaPipe takes about 1.3 s per video, hundreds of times the model. The pose report
estimated 2 to 5 s per video; our machine is at the fast end of that. Decoding the video is the largest cost of the
frames pipeline (the loader reads the whole video before sampling 16 frames). Caveat: one video, one machine; the
MediaPipe time depends somewhat on the content of the video.

### 6.3 Errors

[TBD: `top_confusions.json` and `per_class.csv` of each model. The pose models confuse boy / girl, election / shoes,
monday / hot, time / teacher; do the frames models make the same mistakes?]

## 7. Discussion

* **Accuracy:** [TBD: is any difference larger than the +-4 point interval? Use the paired test.]
* **Speed:** [TBD: model-only against end-to-end; which pipeline is better for which use (real-time, offline, on a phone).]
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

[TBD, 4-5 sentences answering the research question, with the caveats of Section 7.]

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
