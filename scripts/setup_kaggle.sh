#!/usr/bin/env bash
# Usage (inside a Kaggle cell, Internet must be ON in the notebook settings):
#   !bash scripts/setup_kaggle.sh
#   !bash scripts/setup_kaggle.sh --pose    # + MediaPipe, then restart the kernel
set -e   # stop at the first error

# Small extra packages (torch/torchvision are already installed on Colab/Kaggle).
pip install -q pyyaml opencv-python-headless tabulate
# With --pose: also install MediaPipe (only needed for keypoint extraction).
if [[ "$1" == "--pose" ]]; then
  pip install -q -r requirements-pose.txt
  echo ">>> MediaPipe installed. Restart the kernel (Run -> Restart & clear outputs), then continue."
fi
# Print the torch version and whether a GPU is available.
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
