#!/usr/bin/env bash
# Usage (inside a Colab cell):
#   !bash scripts/setup_colab.sh            # training / evaluation
#   !bash scripts/setup_colab.sh --pose     # + MediaPipe (keypoint extraction), then RESTART the runtime
set -e
pip install -q pyyaml opencv-python-headless tabulate tensorboard scikit-learn pandas matplotlib
if [[ "$1" == "--pose" ]]; then
  pip install -q -r requirements-pose.txt
  echo ">>> MediaPipe installed. Restart the runtime now (Runtime -> Restart session), then continue."
fi
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
