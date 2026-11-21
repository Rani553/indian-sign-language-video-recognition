#!/usr/bin/env bash
# Pipeline A ablations (Member 2): change ONE thing at a time around the baseline.
#
#   baseline : mobilenet_v2, 16 frames, freeze 3 epochs then fine-tune, mean pooling
#   backbone : resnet18, efficientnet_b0
#   frames   : 8 frames (the baseline uses 16)
#   training : CNN frozen the whole run | fine-tuned from epoch 0
#   pooling  : last, max
#
# Usage:
#   bash scripts/run_ablations_cnn.sh                  # everything (long: needs a GPU)
#   bash scripts/run_ablations_cnn.sh backbone frames  # only these groups
#   EXTRA="--tiny --set model.pretrained=false" bash scripts/run_ablations_cnn.sh   # quick CPU check
#
# Each run writes results/<experiment>/ and resumes by itself, so the script can be re-run
# after a Colab disconnect. Compare the runs afterwards with: python -m src.evaluation.compare
set -e

CONFIG="${CONFIG:-configs/cnn_lstm.yaml}"
EXTRA="${EXTRA:-}"          # extra arguments passed to every run
GROUPS_TO_RUN=("$@")
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && GROUPS_TO_RUN=(baseline backbone frames training pooling)

run() {                      # run <experiment name> <--set overrides...>
  local name="$1"; shift
  echo; echo "=== $name ==="
  python -m src.training.train_cnn_lstm --config "$CONFIG" $EXTRA --set experiment="$name" "$@"
}

for group in "${GROUPS_TO_RUN[@]}"; do
  case "$group" in
    baseline)
      run cnn_lstm_mobilenet_v2 ;;
    backbone)
      run cnn_lstm_resnet18        model.backbone=resnet18
      run cnn_lstm_efficientnet_b0 model.backbone=efficientnet_b0 ;;
    frames)
      run cnn_lstm_mobilenet_v2_8f frames.num_frames=8 ;;
    training)
      run cnn_lstm_mobilenet_v2_frozen   model.freeze_backbone=true  model.unfreeze_epoch=null
      run cnn_lstm_mobilenet_v2_finetune model.freeze_backbone=false model.unfreeze_epoch=null ;;
    pooling)
      run cnn_lstm_mobilenet_v2_pool_last model.pooling=last
      run cnn_lstm_mobilenet_v2_pool_max  model.pooling=max ;;
    *)
      echo "Unknown group '$group' (use: baseline backbone frames training pooling)"; exit 1 ;;
  esac
done
