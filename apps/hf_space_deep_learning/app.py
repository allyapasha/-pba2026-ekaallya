from __future__ import annotations

import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np


APP_DIR = Path(__file__).resolve().parent
NUMERIC_FEATURE_COLUMNS = ["text_length_words", "engagement_total", "hashtag_count"]
ARTIFACT_FILENAMES = {
    "model": "mlp_model.pkl",
    "vectorizer": "tfidf_vectorizer.pkl",
    "label_encoder": "label_encoder.pkl",
    "scaler": "scaler.pkl",
}

LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
}

SLANG_DICT = {
    "anj": "anjing",
    "anjg": "anjing",
    "anjr": "anjing",
    "anjir": "anjing",
    "anjer": "anjing",
    "ajg": "anjing",
    "gblk": "goblok",
    "gblg": "goblok",
    "goblog": "goblok",
    "bgo": "bego",
    "bngst": "bangsat",
    "bgst": "bangsat",
    "kntl": "kontol",
    "mmk": "memek",
    "jnck": "jancok",
    "jncok": "jancok",
    "jncuk": "jancok",
    "tll": "tolol",
    "tlol": "tolol",
    "bdsm": "bodoh",
    "bdh": "bodoh",
    "gw": "gue",
    "gua": "gue",
    "lu": "lo",
    "elu": "lo",
    "lo": "lo",
    "loe": "lo",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "g": "tidak",
    "tdk": "tidak",
    "gk": "tidak",
    "kyk": "kayak",
    "kek": "kayak",
    "emg": "emang",
    "emng": "emang",
    "bgt": "banget",
    "bngt": "banget",
    "bgtt": "banget",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "dah": "sudah",
    "blm": "belum",
    "blom": "belum",
    "yg": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "sm": "sama",
    "sma": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "org": "orang",
    "ornag": "orang",
    "krn": "karena",
    "krna": "karena",
    "jgn": "jangan",
    "jng": "jangan",
    "bkn": "bukan",
    "gpp": "tidak apa-apa",
    "otw": "on the way",
    "btw": "by the way",
    "cmn": "cuman",
    "lg": "lagi",
    "lgi": "lagi",
    "aja": "saja",
    "aj": "saja",
    "bs": "bisa",
    "bsa": "bisa",
    "dr": "dari",
    "dri": "dari",
    "utk": "untuk",
    "trs": "terus",
    "trus": "terus",
    "msh": "masih",
    "masi": "masih",
    "jd": "jadi",
    "jdi": "jadi",
    "skrg": "sekarang",
    "skrng": "sekarang",
}


def resolve_artifact_path(filename: str) -> Path:
    candidate = APP_DIR / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{filename} tidak ditemukan di folder Space.")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            prev_is_alpha = i > 0 and text[i - 1].isalpha()
            next_is_alpha = i < len(text) - 1 and text[i + 1].isalpha()
            result.append(LEETSPEAK_MAP[char] if prev_is_alpha or next_is_alpha else char)
        else:
            result.append(char)
    text = "".join(result)
    text = " ".join(SLANG_DICT.get(word, word) for word in text.split())
    text = re.sub(r"[^a-z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


MODEL = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["model"]))
VECTORIZER = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["vectorizer"]))
LABEL_ENCODER = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["label_encoder"]))
SCALER = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["scaler"]))


def predict_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {"error": 1.0}
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"empty_after_cleaning": 1.0}

    text_features = VECTORIZER.transform([cleaned_text]).toarray()
    numeric_array = np.array([[len(cleaned_text.split()), 0, 0]], dtype=float)
    scaled_numeric = SCALER.transform(numeric_array)
    model_input = np.hstack([text_features, scaled_numeric])
    probabilities = MODEL.predict_proba(model_input)[0]
    labels = LABEL_ENCODER.inverse_transform(np.arange(len(probabilities)))
    return {label: float(score) for label, score in zip(labels, probabilities)}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Masukkan teks", lines=4),
    outputs=gr.Label(label="Prediksi sentimen", num_top_classes=3),
    title="Sentiment Analysis Indonesian - Deep Learning",
    description=(
        "Versi eksperimen deep learning ringan memakai MLPClassifier. "
        "Output tetap tiga kelas: positive, negative, neutral."
    ),
    examples=[
        ["Saya sangat puas dengan produk ini"],
        ["Pelayanannya buruk dan mengecewakan"],
        ["Produk ini biasa saja"],
    ],
    cache_examples=False,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
