# Pipeline A: raw RGB frames (CNN + LSTM)

*Draft. Baseline numbers are final for the run described; ablation numbers (Section 5) are to be filled in from `results/*/history.csv` and `test_metrics.json` once the ablation runs finish.*

## 1. Approach

Each clip is turned into a fixed number of RGB frames. A pretrained CNN converts every frame into a feature vector, a (bidirectional) LSTM reads the vectors in time order, and the pooled LSTM output is classified into one of the 50 signs. This is the LRCN design of Donahue et al. (2015).

| Part | Choice |
|---|---|
| Input | 16 frames per clip, 224 x 224 RGB, ImageNet-normalised |
| Spatial features | MobileNetV2 pretrained on ImageNet (torchvision), 1280-d vector per frame |
| Temporal model | 1-layer BiLSTM, hidden size 256, dropout 0.3 |
| Pooling | mean over time of the LSTM outputs |
| Classifier | dropout + linear layer, 50 classes |
| Size | 5.40 M parameters (3.18 M trainable while the CNN is frozen) |

## 2. Data

* **Dataset:** INCLUDE-50, the 50-class subset of INCLUDE (Sridhar et al., 2020). The 50 classes are the official list from the AI4Bharat repository (`label_maps/label_map_include50.json`).
* **Videos used:** 958 (the same total as the official INCLUDE-50 train + test, 766 + 192).
* **Source of the files:** a Kaggle copy of INCLUDE (4,284 videos, 262 classes) instead of the original Zenodo record (4,292 videos); about 8 videos are missing from the copy.
* **Split:** our own stratified split per class, seed 42: 684 train / 137 validation / 137 test. **This is not the official INCLUDE-50 split**, so accuracies are not directly comparable with the published 94.5 %.
* **Preprocessing:** every video is decoded once and cached as 24 evenly spaced frames, letterboxed (padded) to 224 x 224, stored as uint8 arrays. Training never reads the raw videos.
* **Temporal sampling:** training takes one random frame from each of 16 equal time segments; validation and test take 16 evenly spaced frames.

## 3. Training

* **Augmentation (clip-consistent, one random draw per clip):** random crop (side 0.8 to 1.0 of the frame) with rotation up to 10 degrees, brightness / contrast / saturation change of up to 20 %. No horizontal flip, because flipping swaps the dominant hand.
* **Freeze, then fine-tune:** the CNN is frozen for the first 3 epochs (only the LSTM and classifier learn), then unfrozen. Learning rate 1e-3 for the new layers; 1e-4 for the CNN after unfreezing.
* **Optimisation:** AdamW (weight decay 1e-4), 2 warm-up epochs then cosine decay, label smoothing 0.1, gradient clipping 1.0, batch size 8, mixed precision, 30 epochs, early stopping with patience 8 (it did not trigger), seed 42.
* **Hardware and time:** one NVIDIA T4 (Kaggle), about 35 s per epoch, about 18 min for the full run.
* The checkpoint with the best validation accuracy is kept (epoch 25).

## 4. Results (baseline: MobileNetV2, 16 frames, freeze 3 epochs then fine-tune, mean pooling)

| Metric | Value |
|---|---|
| Best validation accuracy | 0.949 (130 / 137), epoch 25 |
| **Test accuracy (top-1)** | **0.927 (127 / 137)** |
| Test top-5 accuracy | 1.000 |
| Test loss (with label smoothing 0.1) | 0.928 |

With 137 test videos one video is 0.73 percentage points, and a normal-approximation 95 % interval around 92.7 % is roughly +/- 4.4 points. Differences of a few points between runs are therefore within noise.

## 5. Ablations

One factor is changed at a time from the baseline. Fill in from the finished runs.

| Experiment | Change | Params (M) | Best val acc | Test acc | s / epoch |
|---|---|---|---|---|---|
| `cnn_lstm_mobilenet_v2` | baseline | 5.40 | 0.949 | 0.927 | ~35 |
| `cnn_lstm_resnet18` | backbone ResNet18 | 12.78 | | | |
| `cnn_lstm_efficientnet_b0` | backbone EfficientNet-B0 | | | | |
| `cnn_lstm_mobilenet_v2_8f` | 8 frames instead of 16 | 5.40 | | | |
| `cnn_lstm_mobilenet_v2_frozen` | CNN frozen the whole run | 5.40 | | | |
| `cnn_lstm_mobilenet_v2_finetune` | CNN trained from epoch 0 | 5.40 | | | |
| `cnn_lstm_mobilenet_v2_pool_last` | last-step pooling | 5.40 | | | |
| `cnn_lstm_mobilenet_v2_pool_max` | max pooling | 5.40 | | | |

*Discussion to write after the runs: which factor mattered, accuracy against model size and speed, and how this compares with the pose-keypoint pipeline (Member 3).*

## 6. Limitations

* One seed per configuration and a small test set (137 videos): small differences are not reliable.
* Our split differs from the official INCLUDE-50 split, and the data comes from a Kaggle copy with about 8 missing videos.
* The checkpoint is selected on the validation set, so the validation accuracy is slightly optimistic; the test accuracy is the number to report.
* All signers are from the same recording setting (Chennai), so the model has not been tested on other signers or regions.

## References

1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.* ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and Description.* CVPR 2015. https://arxiv.org/abs/1411.4389
3. M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
