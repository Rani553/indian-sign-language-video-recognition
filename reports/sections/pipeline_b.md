# Pipeline B: pose keypoints (Pose-LSTM and Pose-Transformer)

**Project:** Indian Sign Language word recognition from video (INCLUDE-50)
**Part of the project covered here:** Pipeline B, Member 3 (Pragati Chauhan), GitHub issue #3, branch `feature/pose-models`.

> **Scope of this document.** This is the report of **one part of the project only**: the pose-keypoint pipeline. It describes the work of this part: keypoint extraction, feature engineering, the two pose models, their training and the ablations. The other parts of the project (the data pipeline, the raw-frame pipeline, the evaluation and cross-pipeline comparison, the demo) are done by other team members and are **not** reported here. This part uses from them the shared INCLUDE-50 dataset and split, the generic trainer and the checkpoint format.

## 1. Approach

Instead of looking at pixels, Pipeline B looks at the signer's skeleton. MediaPipe Holistic finds the body pose and both hands in every frame; the model only sees these coordinates, so background, clothing and lighting are removed from the input. Two sequence models are compared on the same features.

| Part | Choice |
|---|---|
| Keypoints | MediaPipe Holistic (`mediapipe==0.10.21`, model complexity 1): body pose (33 points) and both hands (21 points each), raw `x, y, z` stored per frame; a part that is not found is stored as NaN |
| Input | 32 frames per clip, 536 features per frame (see below) |
| Pose-LSTM | LayerNorm, linear 536 to 128 (+GELU, dropout), 2-layer BiLSTM (hidden 128), attention pooling over time, dropout, linear classifier. 0.76 M parameters |
| Pose-Transformer | LayerNorm, linear 536 to 128, learnable CLS token and position embeddings, 3 pre-norm encoder layers (4 heads, feed-forward 256, GELU), CLS output, LayerNorm, linear classifier. 0.48 M parameters |

**Features (536 per frame).** Computed from the raw keypoints in five steps (`src/features/keypoint_features.py`):

1. *Gap filling.* MediaPipe misses hands in many frames; missing frames are filled by linear interpolation (the nearest detection at the ends of a clip, zeros if a hand is never found).
2. *Normalisation.* The origin is the midpoint of the shoulders and the unit is the shoulder width (median over the clip), so the position and size of the signer do not matter.
3. *Features per frame.* Upper-body pose (nose, shoulders, elbows, wrists: 21 values); for each hand its normalised coordinates (63) and its shape relative to the wrist, divided by hand size (60); the frame-to-frame difference (velocity) of all of these; and one hand-presence flag per hand (1 = MediaPipe found the hand in that frame). In total (21 + 2 x 123) x 2 + 2 = 536.
4. *Time resampling.* The 24 cached frames are interpolated to 32.
5. *Augmentation (training only).* Random zoom (+-10 %), rotation (+-8 degrees), shift (+-0.1 shoulder width), Gaussian noise (0.01), time jitter, and with probability 0.2 per hand a short span of frames is dropped and re-interpolated, which mimics MediaPipe misses. Mirroring is off by default (it swaps the dominant hand) and is tested as an ablation.

The models follow the interface agreed in the team (`forward(x) -> logits`, batch `x` of shape `(B, T, F)`), so the shared trainer and the checkpoint format work unchanged.

## 2. Data

* **Dataset and split:** INCLUDE-50, the 50-class subset of INCLUDE (Sridhar et al., 2020), with the shared split prepared by the data pipeline: our own stratified split, seed 42, 684 train / 137 validation / 137 test videos (958 in total). This is **not the official INCLUDE-50 split**, so the accuracies below are not comparable with published numbers.
* **Where the keypoints come from:** the raw videos were not available to this part. MediaPipe was run on the 24 cached frames per video that the data pipeline provides (letterboxed to 224 x 224). The signer is small in these frames, and the 24 frames are spread over the clip rather than consecutive, so MediaPipe runs in image mode (no tracking between frames).
* **Detection rate (all 958 videos, share of frames in which the part was found):** body 99.1 %, left hand 66.9 %, right hand 55.2 %. In a 50-video check, at least one hand was found in 75.7 % of frames and both hands in 43.0 %; 7 of 50 videos had a hand in fewer than half of their frames, none had no hand at all. Part of this is expected (hands rest at the sides at the start and end of a sign, and some signs use one hand), but part is probably caused by the small hands in the 224 x 224 frames. Upscaling the frames by 2 did not help (74.7 %).
* **Reproducibility of the keypoints:** MediaPipe keeps internal state between calls, so the hand coordinates of a video differed by 0.005 to 0.03 (in image units) depending on which video was processed before it; body landmarks and the found / not found pattern were identical. The keypoints used for training were extracted before this was noticed. The extractor now resets MediaPipe before every video, which makes the result independent of the order (checked on two videos); the models were not retrained. As a check of the inference path, the 137 validation videos were pushed through `KeypointExtractor` (cached frames, then MediaPipe with reset, features and the Pose-Transformer checkpoint): 131 / 137 correct (0.956), the same as with the saved keypoints.

## 3. Training

* **Optimisation:** AdamW (weight decay 0.01), linear warm-up (3 epochs for the LSTM, 5 for the Transformer) then cosine decay, label smoothing 0.1, gradient clipping 1.0, batch size 32, at most 100 epochs, early stopping with patience 20, seed 42.
* **Learning rate:** 1e-3 for the Pose-LSTM, 5e-4 for the Pose-Transformer. Dropout 0.3 and 0.2. No hyper-parameter search was done.
* **Hardware and time:** a Colab GPU runtime (the GPU model was not logged): about 0.7 to 1.1 s per epoch, 1 to 2 minutes per run. A CPU laptop took 2 to 3 s per epoch in an earlier check.
* The checkpoint with the best validation accuracy is kept. The test set was evaluated once, on the two baseline checkpoints only, after all training was finished.

## 4. Results (baselines: pose + both hands, wrist-relative hand shape, velocity, augmentation, 32 frames)

| Metric | Pose-LSTM | Pose-Transformer |
|---|---|---|
| Parameters | 0.76 M | 0.48 M |
| Checkpoint file | 3.0 MB | 2.0 MB |
| Best validation accuracy | 0.934 (128 / 137), epoch 42 | 0.956 (131 / 137), epoch 66 |
| **Test accuracy (top-1)** | **0.883 (121 / 137)** | **0.934 (128 / 137)** |
| Test top-5 accuracy | 0.978 | 0.985 |
| CPU time per video, model only (batch 1) | 2.3 ms | 1.3 ms |

With 137 test videos one video is 0.73 percentage points, and a normal-approximation 95 % interval is roughly +-5 points for 88 % and +-4 points for 93 %. The Transformer is 7 videos ahead of the LSTM on the test set and 3 videos ahead on validation; the direction is the same, but with one seed per model and these interval widths we regard the advantage as likely but not established.

**Errors.** The LSTM misclassified 16 test videos and the Transformer 9; four confusions are shared: boy / girl, election / shoes, monday / hot and time / teacher (the Transformer confuses boy and girl twice). Several of these pairs differ mainly in hand shape, which is the information MediaPipe finds least reliably in our frames.

## 5. Ablations

One factor is changed at a time from the baseline; everything else is identical (same split, seed and schedule). Each configuration was run once. Test accuracy was not measured for the ablations, to keep the test set for the two baseline models. The raw table is in `pipeline_b_ablation.csv`.

| Experiment | Change | F | Params (M) | Best val acc | Val top-5 | Epochs run | s / epoch |
|---|---|---|---|---|---|---|---|
| `pose_lstm` | baseline | 536 | 0.759 | 0.934 | 0.993 | 64 | 1.07 |
| `pose_lstm_hands_only` | no body pose | 494 | 0.753 | 0.942 | 0.971 | 84 | 0.95 |
| `pose_lstm_body_only` | no hands | 42 | 0.694 | 0.942 | 0.993 | 89 | 0.82 |
| `pose_lstm_no_velocity` | no velocity features | 269 | 0.724 | 0.927 | 0.978 | 48 | 1.00 |
| `pose_lstm_no_aug` | no augmentation | 536 | 0.759 | 0.942 | 0.978 | 53 | 0.73 |
| `pose_lstm_16f` | 16 frames | 536 | 0.759 | 0.920 | 1.000 | 44 | 0.99 |
| `pose_lstm_mirror` | mirror, p = 0.5 | 536 | 0.759 | 0.927 | 0.978 | 78 | 1.04 |
| `pose_transformer` | baseline | 536 | 0.482 | 0.956 | 1.000 | 88 | 1.10 |
| `pose_transformer_hands_only` | no body pose | 494 | 0.477 | 0.927 | 0.956 | 75 | 1.10 |
| `pose_transformer_body_only` | no hands | 42 | 0.418 | 0.934 | 0.993 | 86 | 0.91 |
| `pose_transformer_no_velocity` | no velocity features | 269 | 0.448 | 0.942 | 0.993 | 61 | 1.04 |
| `pose_transformer_no_aug` | no augmentation | 536 | 0.482 | 0.942 | 0.993 | 73 | 0.77 |
| `pose_transformer_16f` | 16 frames | 536 | 0.482 | 0.949 | 1.000 | 65 | 1.01 |
| `pose_transformer_mirror` | mirror, p = 0.5 | 536 | 0.482 | 0.942 | 0.993 | 100 | 1.03 |

Validation accuracy in videos (of 137): LSTM baseline 128, hands only 129, body only 129, no velocity 127, no augmentation 129, 16 frames 126, mirror 127; Transformer baseline 131, hands only 127, body only 128, no velocity 129, no augmentation 129, 16 frames 130, mirror 129.

### Discussion

**No ablation is distinguishable from the baseline.** All 14 runs lie between 126 and 131 correct videos. The standard deviation of a validation accuracy of about 94 % on 137 videos is already about 3 videos, so every difference in the table is within one standard deviation of the baseline. We therefore draw no ranking from this table, and we do not claim that any single component (hands, body, velocity, augmentation, mirroring, 32 instead of 16 frames) is necessary or harmful for our data size.

What the table does show, and what is worth stating:

* **Either part of the input is sufficient on its own.** Hands only (129 and 127 videos) and body only (129 and 128 videos) are about as accurate as both together (128 and 131). A plausible but untested explanation: the body-only features still contain the elbow and wrist positions over time, and the hand-only features contain where the hands are (they are normalised by the shoulders, not by themselves), so both describe the coarse arm movement, which already separates most of the 50 signs. Hand shape matters for the remaining confusions (boy / girl), where all variants probably struggle; we did not break the errors down per variant.
* **Half the frames (16) cost nothing measurable** (126 and 130 videos against 128 and 131), which suggests the 24 cached frames carry more temporal detail than needed. The time per epoch hardly changed (0.99 s against 1.07 s), so there is no practical speed gain at this size.
* **Augmentation did not show a benefit.** Runs without it reach 129 videos for both models. Our augmentation was not tuned; the training curves of the baseline LSTM show a gap between training accuracy (about 0.98) and validation accuracy (0.90 to 0.93), so some over-fitting is present and a stronger or different augmentation might still help.
* **The mirror run of the Transformer used all 100 epochs** (best at epoch 80) while the others stopped early, so it may have benefited from more training; its result is not clearly different from the baseline.

## 6. Size and speed of this pipeline

* **Models:** 0.48 M (Transformer) and 0.76 M (LSTM) parameters; checkpoints of 2.0 and 3.0 MB. Once the keypoints exist, one video takes 1.3 to 2.3 ms on a laptop CPU (model only), and a training epoch about 1 s on a Colab GPU.
* **The keypoint step dominates.** On a laptop CPU MediaPipe took about 2.3 s per video (24 frames, one process) in an isolated measurement, plus 0.15 s for the reset; the complete path from cached frames to a prediction (keypoints, features, model) took 5.4 s per video on average over the 137 validation videos. The two figures were measured at different times on a laptop that was also used for other work, so they differ by more than the reset alone explains; we report both and take about 2 to 5 s per video as the realistic range. Either way this is roughly a thousand times the model time or more. Extracting the keypoints of all 958 videos took about 20 minutes with 6 processes. The model times above therefore describe the pipeline only after this step; the cost of a prediction from a raw video is dominated by MediaPipe.
* A measurement of the complete path from a raw video to a prediction, on the same hardware as the other pipelines, is part of the project-level benchmark and is not reported here.

## 7. Limitations

* One seed per configuration and small validation and test sets (137 videos each): differences of up to about three videos are not reliable, which is why the ablations are inconclusive.
* Keypoints come from 224 x 224 letterboxed frames, not from the original videos; the hands were found in only about 55 to 67 % of the frames. Keypoints from the raw videos would very probably give more reliable hand features and could change both the accuracy and the ablation results.
* The 24 cached frames are not consecutive, so no motion tracking is possible in MediaPipe; the velocity features are differences between frames that are spread evenly over the whole video (several original frames apart), so they describe coarse motion only.
* The keypoints used for training were extracted without resetting MediaPipe between videos (small, order-dependent differences in hand coordinates, see Section 2); the models were not retrained after the fix.
* Face landmarks were not used. All signers are from the same recording setting, so generalisation to other signers or regions is untested.
* The split is our own, not the official INCLUDE-50 split, so the results cannot be compared with published numbers.
* **Possible overlap between training and test sessions (suspected, not proven).** We assume that consecutive file numbers of one class are takes from the same recording session; the dataset has no signer id to confirm this. With this assumption, 109 of the 137 test videos have a training video of the same class within one file number, and 127 within two. A random split like ours may therefore be easier than recognising a new session or a new signer, and the accuracies above may be higher than they would be for new signers. A split by recording session would be needed to measure the size of this effect, and all models would have to be retrained for it.
* The checkpoint is selected on the validation set, so the validation accuracy is slightly optimistic; the test accuracy is the number to report. The test set was used only for the two baseline checkpoints.
* No hyper-parameter tuning and no repeated runs; the Pose-LSTM versus Pose-Transformer difference should be confirmed with several seeds.

## Deliverables of this part

* Code: `src/features/` (`extract_keypoints.py`, `keypoint_features.py`, `keypoint_io.py`), `src/data/keypoint_dataset.py`, `src/models/pose_lstm.py`, `src/models/pose_transformer.py`, `src/training/train_pose.py`, `configs/pose_lstm.yaml`, `configs/pose_transformer.yaml`, `scripts/run_ablations_pose.sh`, tests in `tests/`.
* `KeypointExtractor` (`src/features/extract_keypoints.py`): turns a raw video into keypoints with the same preprocessing as training, for use by other parts of the project.
* Best Pose-LSTM and Pose-Transformer checkpoints (shared outside Git) and the ablation table `pipeline_b_ablation.csv`.
* Colab notebook `notebooks/colab/m3_pose_models.ipynb`.

## Reproducing

```bash
python -m src.features.extract_keypoints          # needs requirements-pose.txt; about 20 min for 958 videos
python -m src.training.train_pose --config configs/pose_lstm.yaml
python -m src.training.train_pose --config configs/pose_transformer.yaml
bash scripts/run_ablations_pose.sh                # the 12 ablation runs (plus both baselines)
```

## References

1. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.* ACM Multimedia 2020. https://doi.org/10.1145/3394171.3413528
2. C. Lugaresi et al. *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172, 2019.
3. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
