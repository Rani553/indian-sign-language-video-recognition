# Pipeline A: raw RGB frames (CNN + LSTM)

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
* **Optimisation:** AdamW (weight decay 1e-4), 2 warm-up epochs then cosine decay, label smoothing 0.1, gradient clipping 1.0, batch size 8, mixed precision, at most 30 epochs, early stopping with patience 8, seed 42.
* **Hardware and time:** one NVIDIA T4 (Kaggle), about 35 s per epoch, about 18 min for the baseline run.
* The checkpoint with the best validation accuracy is kept; the test set is used once per run, after training.

## 4. Results (baseline: MobileNetV2, 16 frames, freeze 3 epochs then fine-tune, mean pooling)

| Metric | Value |
|---|---|
| Best validation accuracy | 0.949 (130 / 137), epoch 25 |
| **Test accuracy (top-1)** | **0.927 (127 / 137)** |
| Test top-5 accuracy | 1.000 |
| Test loss (with label smoothing 0.1) | 0.928 |

With 137 test videos one video is 0.73 percentage points, and a normal-approximation 95 % interval around 92.7 % is roughly +/- 4.4 points. Differences of a few points between runs are therefore within noise.

## 5. Ablations

One factor is changed at a time from the baseline; everything else is identical (same split, seed and schedule). Each configuration was run once. The raw table is in `pipeline_a_ablation.csv`.

| Experiment | Change | Params (M) | Best val acc | Test acc | Test top-5 | Epochs run | s / epoch |
|---|---|---|---|---|---|---|---|
| `cnn_lstm_mobilenet_v2` | baseline (MobileNetV2, 16 frames, freeze 3 ep, mean pool) | 5.40 | 0.949 | 0.927 | 1.000 | 30 | 35 |
| `cnn_lstm_resnet18` | backbone ResNet18 | 12.78 | 0.883 | 0.861 | 0.993 | 30 | 39 |
| `cnn_lstm_efficientnet_b0_run1` | backbone EfficientNet-B0 (checkpoint lost) | 7.18 | 0.964 | 0.934 | 1.000 | 28 | 42 |
| `cnn_lstm_efficientnet_b0` | the same run repeated (this checkpoint is shared) | 7.18 | 0.971 | 0.978 | 1.000 | 29 | 41 |
| `cnn_lstm_mobilenet_v2_8f` | 8 frames instead of 16 | 5.40 | 0.905 | 0.912 | 1.000 | 30 | 18 |
| `cnn_lstm_mobilenet_v2_frozen` | CNN frozen the whole run | 5.40 | 0.467 | 0.453 | 0.927 | 30 | 36 |
| `cnn_lstm_mobilenet_v2_finetune` | CNN trained from epoch 0 | 5.40 | 0.942 | 0.920 | 0.993 | 28 | 37 |
| `cnn_lstm_mobilenet_v2_pool_last` | last-step pooling | 5.40 | 0.759 | 0.745 | 0.993 | 30 | 38 |
| `cnn_lstm_mobilenet_v2_pool_max` | max pooling | 5.40 | 0.942 | 0.934 | 0.993 | 30 | 39 |

Test accuracy in videos (of 137): baseline 127, EfficientNet-B0 128 (first run) and 134 (repeated run), max pooling 128, fine-tune from start 126, 8 frames 125, ResNet18 118, last-step pooling 102, frozen CNN 62.

### Discussion

**Differences that are clearly larger than the noise**

* **A frozen CNN does not work.** Keeping the ImageNet features fixed gives 0.453 test accuracy against 0.927 for the baseline (about 47 points). ImageNet features alone do not separate these signs; the CNN has to adapt to the data.
* **Last-step pooling is much worse than mean or max pooling** (0.745 against 0.927 and 0.934). Its validation accuracy was still rising slowly at epoch 30, so part of the gap may be slower convergence rather than a hard limit.

**Differences that are small or not conclusive**

* **Backbone.** The first EfficientNet-B0 run (0.934, 128 videos) lost its checkpoint when the Kaggle session ended, so the same configuration was run again with the same seed (0.978, 134 videos; best validation accuracy 0.971 against 0.964). Two runs of one configuration therefore differ by six test videos, which is larger than most differences in this table, and we do not claim that EfficientNet-B0 is better than MobileNetV2 (0.927, 127 videos). ResNet18 is lower (0.861, 118 videos), but its validation accuracy was still increasing at epoch 30, so it was probably not converged; a longer run was not done. EfficientNet-B0 reaches a training accuracy of 0.99, so it overfits more than MobileNetV2.
* **Pooling.** Max (0.934) and mean (0.927) pooling differ by one video: no evidence of a difference.
* **Freeze schedule.** Fine-tuning from the first epoch (0.920) and freezing the CNN for 3 epochs first (0.927) differ by one video. Freezing for the first epochs gave no measurable benefit on this dataset; fine-tuning itself is what matters.
* **Number of frames.** 8 frames cost about 1.5 points of test accuracy (0.912 against 0.927, two videos) and 4.4 points of validation accuracy, and halve the time per epoch (18 s against 35 s). Within the noise of a single run this suggests that 8 frames are nearly as accurate at about half the cost, which would be worth confirming with more seeds.

**Accuracy against size and speed.** Parameter counts range from 5.40 M (MobileNetV2) to 12.78 M (ResNet18) without a matching accuracy gain, so the lightest model is the sensible default. The time per epoch was almost the same for all backbones (35 to 42 s) and for the frozen run, but half as long with 8 frames. This suggests that training time here is limited by loading and augmenting frames on the CPU rather than by the GPU; we did not profile this. Inference speed on CPU and GPU is measured separately in the comparison with the pose-keypoint pipeline.

**Model for the comparison with Pipeline B.** The best checkpoint by validation accuracy is the repeated EfficientNet-B0 run (val 0.971, test 0.978); it was chosen on validation accuracy, not on the test result. The MobileNetV2 baseline (test 0.927) has fewer parameters, and the first EfficientNet-B0 run (test 0.934) shows that the gap between the two is within the run-to-run variation. Both checkpoints are provided.

## 6. Limitations

* One seed per configuration and a small test set (137 videos): differences of one to three videos are not reliable.
* Our split differs from the official INCLUDE-50 split, and the data comes from a Kaggle copy with about 8 missing videos.
* The checkpoint is selected on the validation set, so the validation accuracy is slightly optimistic; the test accuracy is the number to report.
* ResNet18 and last-step pooling were still improving when training stopped at 30 epochs, so their results understate what longer training could reach.
* All signers are from the same recording setting (Chennai), so the model has not been tested on other signers or regions.

## References

1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.* ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. J. Donahue et al. *Long-term Recurrent Convolutional Networks for Visual Recognition and Description.* CVPR 2015. https://arxiv.org/abs/1411.4389
3. M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
