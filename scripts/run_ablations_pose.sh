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

# Stop at the first failing run. Otherwise a broken first run would be hidden by thirteen
# "successful" ones that follow it, and the results table would silently have a gap.
set -e

# Values can be overridden from outside (see Usage). ${VAR:-default} means "use VAR if set".
MODELS="${MODELS:-pose_lstm pose_transformer}"
EXTRA="${EXTRA:-}"          # extra arguments passed to every run
GROUPS_TO_RUN=("$@")
# No arguments given: run every group. (A name other than GROUPS is used because bash already
# has a built-in variable called GROUPS.)
[ ${#GROUPS_TO_RUN[@]} -eq 0 ] && GROUPS_TO_RUN=(baseline parts velocity augmentation frames mirror)

run() {                      # run <model> <suffix> <--set overrides...>
  local model="$1" suffix="$2"; shift 2
  # The experiment name decides the results folder. Every ablation gets its own name
  # (pose_lstm_no_aug ...), so runs never overwrite each other and can be resumed one by one.
  # ${suffix:+_$suffix} adds "_<suffix>" only if a suffix was given (the baseline has none).
  local name="$model${suffix:+_$suffix}"
  echo; echo "=== $name ==="
  # $EXTRA is deliberately unquoted so that several words (for example "--tiny --set ...") stay
  # separate arguments; the quoted "$@" keeps each override intact.
  python -m src.training.train_pose --config "configs/$model.yaml" $EXTRA --set experiment="$name" "$@"
}

# Both models get the same ablations, so the two result tables can be compared side by side.
for model in $MODELS; do
  for group in "${GROUPS_TO_RUN[@]}"; do
    case "$group" in
      baseline)
        run "$model" "" ;;
      parts)
        # Which part of the input carries the information? Remove one at a time.
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
        # A typo in a group name must not be skipped silently, so stop with the valid names.
        echo "Unknown group '$group' (use: baseline parts velocity augmentation frames mirror)"; exit 1 ;;
    esac
  done
done
