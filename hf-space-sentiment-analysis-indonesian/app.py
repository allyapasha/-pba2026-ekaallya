import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_FILENAMES = {
    "model": "sentiment_model.pkl",
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


def get_artifact_dirs() -> list[Path]:
    candidates = [
        APP_DIR / "models",
        APP_DIR.parent / "models",
        APP_DIR,
        Path.cwd() / "models",
        Path.cwd(),
    ]
    ordered_candidates = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(resolved)
    return ordered_candidates


def resolve_artifact_path(filename: str) -> Path:
    for directory in get_artifact_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in get_artifact_dirs())
    raise FileNotFoundError(f"{filename} not found. Searched: {searched}")


def load_artifacts():
    model = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["model"]))
    vectorizer = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["vectorizer"]))
    label_encoder = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["label_encoder"]))
    scaler = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["scaler"]))
    return model, vectorizer, label_encoder, scaler


ARTIFACT_LOAD_ERROR = None

try:
    MODEL, VECTORIZER, LABEL_ENCODER, SCALER = load_artifacts()
except Exception as exc:
    MODEL = VECTORIZER = LABEL_ENCODER = SCALER = None
    ARTIFACT_LOAD_ERROR = str(exc)


def normalize_leetspeak(text: str) -> str:
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            prev_is_alpha = i > 0 and text[i - 1].isalpha()
            next_is_alpha = i < len(text) - 1 and text[i + 1].isalpha()
            result.append(LEETSPEAK_MAP[char] if prev_is_alpha or next_is_alpha else char)
        else:
            result.append(char)
    return "".join(result)


def expand_slang(text: str) -> str:
    return " ".join(SLANG_DICT.get(word, word) for word in text.split())


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = normalize_leetspeak(text)
    text = expand_slang(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_feature_frame(cleaned_text: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cleaned_text": [cleaned_text],
            "text_length_words": [len(cleaned_text.split())],
            "engagement_total": [0],
            "hashtag_count": [0],
        }
    )


def predict_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {"error": 1.0}
    if ARTIFACT_LOAD_ERROR:
        return {f"model_unavailable: {ARTIFACT_LOAD_ERROR}": 1.0}

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"empty_after_cleaning": 1.0}

    features = build_feature_frame(cleaned_text)
    text_features = VECTORIZER.transform(features["cleaned_text"]).toarray()
    numeric_features = features[["text_length_words", "engagement_total", "hashtag_count"]]
    scaled_numeric = SCALER.transform(numeric_features)
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


description = (
    "Model klasifikasi sentimen untuk teks berbahasa Indonesia "
    "dengan tiga kelas: positive, negative, dan neutral."
)

if ARTIFACT_LOAD_ERROR:
    description = (
        f"{description}\n\n"
        f"Startup warning: artefak model gagal dimuat. {ARTIFACT_LOAD_ERROR}"
    )


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Masukkan teks",
        placeholder="Contoh: Saya sangat puas dengan produk ini.",
        lines=4,
    ),
    outputs=gr.Label(label="Prediksi sentimen", num_top_classes=3),
    title="Sentiment Analysis Indonesian",
    description=description,
    examples=EXAMPLES,
    cache_examples=False,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
