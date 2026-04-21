import sys
from pathlib import Path

import joblib
import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_system import ARTIFACT_FILENAMES, NUMERIC_FEATURE_COLUMNS, build_inference_frame, clean_text, force_single_thread


def get_artifact_dirs() -> list[Path]:
    candidates = [
        APP_DIR / "models",
        APP_DIR.parent / "models",
        APP_DIR,
        Path.cwd() / "models",
        Path.cwd(),
    ]
    ordered = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def resolve_artifact_path(filename: str) -> Path:
    for directory in get_artifact_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in get_artifact_dirs())
    raise FileNotFoundError(f"{filename} not found. Searched: {searched}")


def load_artifacts():
    model = force_single_thread(joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["model"])))
    vectorizer = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["vectorizer"]))
    label_encoder = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["label_encoder"]))
    scaler = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["scaler"]))
    return model, vectorizer, label_encoder, scaler


ARTIFACT_LOAD_ERROR = None
MODEL = VECTORIZER = LABEL_ENCODER = SCALER = None

try:
    MODEL, VECTORIZER, LABEL_ENCODER, SCALER = load_artifacts()
except Exception as exc:
    ARTIFACT_LOAD_ERROR = str(exc)


def predict_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {"error": 1.0}
    if ARTIFACT_LOAD_ERROR:
        return {f"model_unavailable: {ARTIFACT_LOAD_ERROR}": 1.0}

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"empty_after_cleaning": 1.0}

    features = build_inference_frame(cleaned_text)
    text_features = VECTORIZER.transform(features["cleaned_text"]).toarray()
    scaled_numeric = SCALER.transform(features[NUMERIC_FEATURE_COLUMNS].to_numpy())
    model_input = np.hstack([text_features, scaled_numeric])

    probabilities = MODEL.predict_proba(model_input)[0]
    labels = LABEL_ENCODER.inverse_transform(np.arange(len(probabilities)))
    return {label: float(score) for label, score in zip(labels, probabilities)}


EXAMPLES = [
    ["Saya sangat senang dengan produk ini."],
    ["Pelayanannya buruk dan mengecewakan."],
    ["Produk ini biasa saja, tidak istimewa."],
    ["Saya suka kualitasnya, pengirimannya juga cepat."],
    ["Saya tidak puas dengan hasil pembelian ini."],
]

DESCRIPTION = (
    "Model klasifikasi sentimen 3 kelas dengan pipeline "
    "TF-IDF + fitur numerik sederhana + Random Forest. "
    "Aplikasi ini bukan menggunakan model deep learning."
)
if ARTIFACT_LOAD_ERROR:
    DESCRIPTION = f"{DESCRIPTION}\n\nStartup warning: {ARTIFACT_LOAD_ERROR}"
if gr is None:
    DESCRIPTION = f"{DESCRIPTION}\n\nUI warning: gradio belum terpasang di environment ini."


def build_demo():
    if gr is None:
        raise RuntimeError(
            "gradio belum terpasang. Install dependency dengan 'pip install -r project-ml/app/requirements.txt'."
        )
    return gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(
            label="Masukkan teks",
            placeholder="Contoh: Saya sangat puas dengan produk ini.",
            lines=4,
        ),
        outputs=gr.Label(label="Prediksi sentimen", num_top_classes=3),
        title="Sentiment Analysis Indonesian",
        description=DESCRIPTION,
        examples=EXAMPLES,
        cache_examples=False,
        flagging_mode="never",
    )


demo = build_demo() if gr is not None else None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError(
            "gradio belum terpasang. Jalankan: pip install -r project-ml/app/requirements.txt"
        )
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
