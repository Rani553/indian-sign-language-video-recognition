#!/usr/bin/env bash
# Usage (inside a Colab cell):
#   !bash scripts/setup_colab.sh            # training / evaluation
#   !bash scripts/setup_colab.sh --pose     # + MediaPipe (keypoint extraction), then RESTART the runtime
set -e   # stop at the first error

# Small extra packages (torch/torchvision are already installed on Colab/Kaggle).
pip install -q pyyaml opencv-python-headless tabulate tensorboard scikit-learn pandas matplotlib
# With --pose: also install MediaPipe (only needed for keypoint extraction).
if [[ "$1" == "--pose" ]]; then
  pip install -q -r requirements-pose.txt
  echo ">>> MediaPipe installed. Restart the runtime now (Runtime -> Restart session), then continue."
fi
# Print the torch version and whether a GPU is available.
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
