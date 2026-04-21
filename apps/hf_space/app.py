from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gradio as gr
except ImportError:
    gr = None

from src.sentiment_project.inference import load_classic_artifacts, predict_with_classic_pipeline


ARTIFACT_LOAD_ERROR = None
CLASSIC_ARTIFACTS = None

try:
    CLASSIC_ARTIFACTS = load_classic_artifacts(extra_dirs=[APP_DIR / "models", APP_DIR])
except Exception as exc:
    ARTIFACT_LOAD_ERROR = str(exc)


DESCRIPTION = (
    "Space ini memakai model production klasik: Logistic Regression + TF-IDF + fitur numerik sederhana. "
    "Baseline deep learning tetap dipisahkan sebagai jalur eksperimen."
)
if ARTIFACT_LOAD_ERROR:
    DESCRIPTION = f"{DESCRIPTION}\n\nStartup warning: {ARTIFACT_LOAD_ERROR}"


def predict_sentiment(text: str) -> dict:
    if ARTIFACT_LOAD_ERROR:
        return {f"model_unavailable: {ARTIFACT_LOAD_ERROR}": 1.0}
    return predict_with_classic_pipeline(text, artifacts=CLASSIC_ARTIFACTS)


def build_demo():
    if gr is None:
        raise RuntimeError("gradio belum terpasang.")
    return gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(label="Masukkan teks", lines=4),
        outputs=gr.Label(label="Prediksi sentimen", num_top_classes=3),
        title="Sentiment Analysis Indonesian",
        description=DESCRIPTION,
        examples=[
            ["Saya sangat puas dengan produk ini"],
            ["Pelayanannya buruk dan mengecewakan"],
            ["Produk ini biasa saja"],
        ],
        cache_examples=False,
        flagging_mode="never",
    )


demo = build_demo() if gr is not None else None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("gradio belum terpasang. Install dependency dari requirements.txt")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

