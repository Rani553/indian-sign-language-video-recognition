#!/usr/bin/env bash
# Pipeline B ablations (Member 3): change ONE thing at a time around the baseline, for both models.
#
#   baseline     : pose + both hands, wrist-relative hand shape, velocity, augmentation, 32 frames
#   parts        : hands only | body only
#   velocity     : no velocity features
#   augmentation : no augmentation
#   frames       : 16 frames (the baseline uses 32)
#   mirror       : mirror augmentation (left <-> right) with probability 0.5
#
# Usage:
#   bash scripts/run_ablations_pose.sh                       # everything, both models
#   bash scripts/run_ablations_pose.sh parts velocity        # only these groups
#   MODELS="pose_lstm" bash scripts/run_ablations_pose.sh    # only one model
#   EXTRA="--tiny" bash scripts/run_ablations_pose.sh        # quick CPU check
#
# Each run writes results/<experiment>/ and resumes by itself, so the script can be re-run
# after a Colab disconnect. Compare the runs afterwards with: python -m src.evaluation.compare
set -e

MODELS="${MODELS:-pose_lstm pose_transformer}"
EXTRA="${EXTRA:-}"          # extra arguments passed to every run
GROUPS_TO_RUN=("$@")
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && GROUPS_TO_RUN=(baseline parts velocity augmentation frames mirror)

run() {                      # run <model> <suffix> <--set overrides...>
  local model="$1" suffix="$2"; shift 2
  local name="$model${suffix:+_$suffix}"
  echo; echo "=== $name ==="
  python -m src.training.train_pose --config "configs/$model.yaml" $EXTRA --set experiment="$name" "$@"
}

for model in $MODELS; do
  for group in "${GROUPS_TO_RUN[@]}"; do
    case "$group" in
      baseline)
        run "$model" "" ;;
      parts)
        run "$model" hands_only keypoints.parts=[left_hand,right_hand]
        run "$model" body_only  keypoints.parts=[pose] ;;
      velocity)
        run "$model" no_velocity keypoints.use_velocity=false ;;
      augmentation)
        run "$model" no_aug augment.enabled=false ;;
      frames)
        run "$model" 16f keypoints.num_frames=16 ;;
      mirror)
        run "$model" mirror augment.mirror=0.5 ;;
      *)
        echo "Unknown group '$group' (use: baseline parts velocity augmentation frames mirror)"; exit 1 ;;
    esac
  done
done
