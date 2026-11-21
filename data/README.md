# Data

Nothing in this folder is committed to Git (see `.gitignore`). On Colab/Kaggle the data
lives outside the repository (Drive or `/kaggle/working`); locally it defaults to `./data`.

## Dataset

**INCLUDE** (Sridhar et al., ACM Multimedia 2020) - isolated Indian Sign Language words,
recorded by deaf students, grouped into categories (Adjectives, Animals, ...). Hosted on
Zenodo (record id in `configs/data.yaml` → `dataset.zenodo_record`; verify it on zenodo.org).
**INCLUDE-50** is a 50-class subset defined by the authors.

> The dataset is large. Use `python -m src.data.download --list` to see zip sizes before downloading.
> The downloader handles one zip at a time and deletes it after extracting only the needed classes.

## Choosing the 50 classes
* **Official list (preferred):** put the 50 class names (one per line, folder name without the
  number, e.g. `loud`) in `configs/include50_classes.txt`. Check the INCLUDE paper's supplementary
  material / the AI4Bharat INCLUDE GitHub repository.
* **Fallback:** if that file does not exist, `src.data.split` picks the 50 classes with the most
  videos (`dataset.subset_strategy`) and saves them to `processed/splits/classes.txt`.
  **Say so in the report** - results are then not directly comparable with the paper.

## Layout after preparation
```
<data_root>/
├── raw/<Category>/<N. word>/<video>.MOV     # extracted INCLUDE videos
└── processed/                               # <- share THIS with the team
    ├── splits/   metadata.csv, train/val/test.csv, label_map.json, classes.txt
    ├── frames/   <video_id>.npy   (24, 224, 224, 3) uint8         - Member 2 input
    └── keypoints/<video_id>.npz   raw MediaPipe landmarks         - Member 3 input
```

## Steps (Member 1, Member 3 for keypoints)
```bash
python -m src.data.download --classes configs/include50_classes.txt
python -m src.data.split
python -m src.features.extract_frames
python -m src.features.extract_keypoints     # needs requirements-pose.txt, CPU is fine
```

## Sharing
* **Kaggle:** create a Dataset from `processed/` (e.g. `isl-include50-processed`) and attach it;
  run commands with `--set paths.processed_dir=/kaggle/input/isl-include50-processed`.
* **Colab:** share the Drive folder `isl_data`; each member adds a shortcut to *My Drive*.

## Test without the real data
```bash
python scripts/make_dummy_data.py --data-root /tmp/isl_dummy --fake-keypoints
export ISL_DATA_ROOT=/tmp/isl_dummy
```
