"""Streamlit demo: upload a sign video, see what each trained model predicts and how long it took.

    streamlit run app/demo.py

It lists every ``<results_root>/*/best.pt``. Pick one or more models (for example one frames model and
one pose model) to compare their predictions and speed on the same video. The pose models need
MediaPipe 0.10.21 (requirements-pose.txt); without it they show a message instead of a prediction.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports work from any folder

import pandas as pd
import streamlit as st

from src.utils.config import load_config, resolve_paths

st.set_page_config(page_title="ISL word recognition", layout="wide")

DEFAULT_ROOT = resolve_paths(load_config("configs/data.yaml"))["results_root"]
PIPELINE = {"frames": "Pipeline A (raw frames)", "keypoints": "Pipeline B (pose keypoints)"}


@st.cache_resource(show_spinner="Loading model ...")
def get_predictor(path: str, device: str):
    from src.inference.predictor import Predictor
    return Predictor(path, device)


def find_checkpoints(root: Path) -> dict[str, Path]:
    return {p.parent.name: p for p in sorted(root.glob("*/best.pt"))} if root.exists() else {}


st.title("Indian Sign Language word recognition")
st.caption("INCLUDE-50: 50 signs, one word per clip. Compares raw-frame models (CNN + LSTM) with pose-keypoint models.")

with st.sidebar:
    root = Path(st.text_input("Results folder", str(DEFAULT_ROOT)))
    checkpoints = find_checkpoints(root)
    chosen = st.multiselect("Models", list(checkpoints), default=list(checkpoints)[:1])
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])
    top_k = st.slider("Top-k", 1, 10, 5)

if not checkpoints:
    st.warning(f"No checkpoints found in `{root}` (expected `<experiment>/best.pt`).")
    st.stop()

upload = st.file_uploader("Sign video", type=["mov", "mp4", "avi", "mkv", "webm", "m4v"])
if upload is None:
    st.info("Upload a short video of one sign.")
    st.stop()

st.video(upload)
with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as tmp:
    tmp.write(upload.getbuffer())
video_path = Path(tmp.name)

if not chosen:
    st.info("Choose at least one model in the sidebar.")
    st.stop()

cols = st.columns(len(chosen))
for col, name in zip(cols, chosen):
    with col:
        st.subheader(name)
        try:
            predictor = get_predictor(str(checkpoints[name]), device)
            with st.spinner("Predicting ..."):
                top, timing = predictor.predict(video_path, k=top_k, return_timing=True)
        except Exception as e:  # e.g. MediaPipe missing for a pose model, or a broken video
            st.error(f"{type(e).__name__}: {e}")
            continue
        st.caption(PIPELINE.get(predictor.input_type, predictor.input_type) + f" | device: {predictor.device}")
        st.metric("Prediction", top[0][0], f"{top[0][1]:.1%} confidence")
        st.bar_chart(pd.DataFrame(top, columns=["sign", "probability"]).set_index("sign"))
        st.table(pd.DataFrame({"stage": ["decode + resize", "features", "model", "total"],
                               "ms": [round(timing[k], 1) for k in
                                      ("decode_resize_ms", "features_ms", "model_ms", "total_ms")]}))
        st.caption("`features` is the CNN input for frames models and MediaPipe + feature engineering for pose models.")
video_path.unlink(missing_ok=True)
