"""
app.py — Gradio App untuk Sentiment Analysis (Positive, Negative, Neutral)
==========================================================================
Deploy di Hugging Face Spaces.
Model: PyCaret Classification Pipeline (.pkl)
Mengikuti struktur dari deteksi-toksisitas-chat dengan adaptasi untuk sentiment analysis.
"""

print("Starting app initialization...")
import re

print("Importing Gradio...")
import gradio as gr

print("Importing Pandas...")
import pandas as pd

print("Importing PyCaret...")
import warnings

from pycaret.classification import load_model, predict_model

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# 📦 LOAD MODEL
# ══════════════════════════════════════════════
print("Loading Model...")
try:
    model = load_model("sentiment_pycaret_best")
    print("✅ Model Successfully Loaded!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model with default name: {e}")
    print("Attempting alternative model loading...")
    try:
        model = load_model("sentiment_pycaret_best.pkl")
        print("✅ Model loaded from .pkl file!")
    except Exception as e2:
        print(f"⚠️ Error loading model: {e2}")
        model = None

# ══════════════════════════════════════════════
# 🔤 PREPROCESSING (sama persis dengan training)
# ══════════════════════════════════════════════

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
    # Kata kasar / toxic
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
    # Slang umum
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
    # Gaming terms
    "noob": "pemula",
    "newbie": "pemula",
    "pro": "profesional",
    "gg": "good game",
    "wp": "well played",
    "afk": "away from keyboard",
    "ez": "easy",
    "lag": "lag",
    "dc": "disconnect",
    "bcs": "karena",
}


def normalize_leetspeak(text: str) -> str:
    """Konversi angka/simbol leetspeak ke huruf biasa."""
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            prev_is_alpha = i > 0 and text[i - 1].isalpha()
            next_is_alpha = i < len(text) - 1 and text[i + 1].isalpha()
            if prev_is_alpha or next_is_alpha:
                result.append(LEETSPEAK_MAP[char])
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def expand_slang(text: str) -> str:
    """Ekspansi singkatan & slang gamer ke bentuk lengkap."""
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """Pipeline pembersihan teks (sama persis dengan training)."""
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


# ══════════════════════════════════════════════
# 🎯 FUNGSI PREDIKSI
# ══════════════════════════════════════════════


def predict_sentiment(text: str) -> dict:
    """
    Prediksi label sentimen dari teks input.

    Args:
        text: Teks input untuk diprediksi

    Returns:
        Dictionary {label: confidence} untuk Gradio Label component.
    """
    if not text or not text.strip():
        return {"Error": 1.0}

    if model is None:
        return {"Model Error": 1.0}

    # 1. Bersihkan teks
    cleaned = clean_text(text)

    if not cleaned:
        return {"Teks kosong setelah dibersihkan": 1.0}

    # 2. Buat DataFrame (PyCaret butuh DataFrame sebagai input)
    df_input = pd.DataFrame({"cleaned_text": [cleaned]})

    try:
        # 3. Prediksi menggunakan PyCaret
        result = predict_model(model, data=df_input)

        # 4. Ambil hasil
        if "prediction_label" in result.columns:
            label = result["prediction_label"].iloc[0]
        elif "Label" in result.columns:
            label = result["Label"].iloc[0]
        else:
            # Fallback to last column
            label = str(result.iloc[0, -1])

        if "prediction_score" in result.columns:
            score = result["prediction_score"].iloc[0]
        elif "Score" in result.columns:
            score = result["Score"].iloc[0]
        else:
            score = 1.0

        return {str(label): float(score)}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error during prediction": 1.0}


# ══════════════════════════════════════════════
# 🎨 GRADIO INTERFACE
# ══════════════════════════════════════════════

EXAMPLES = [
    ["Saya sangat senang dengan produk ini! Kualitasnya luar biasa!"],
    ["Terrible service, never coming back here!"],
    ["Harga cukup mahal tetapi kualitasnya standar."],
    ["This is the best day of my life!"],
    ["Layanan pelanggan mereka sangat buruk dan lamban."],
    ["Produk ini biasa saja, tidak istimewa."],
    ["Absolutely fantastic experience, highly recommend!"],
    ["Tidak puas dengan hasil pembelian saya."],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="💬 Masukkan Teks",
        placeholder="Ketik teks untuk analisis sentimen... (contoh: 'Saya sangat menyukai produk ini!')",
        lines=4,
    ),
    outputs=gr.Label(
        label="🎯 Hasil Prediksi Sentimen",
        num_top_classes=3,
    ),
    title="💭 Sentiment Analysis - Indonesian Text",
    description=(
        "Model NLP untuk menganalisis sentimen teks berbahasa Indonesia. "
        "Mengklasifikasikan teks menjadi **Positive** (Positif), **Negative** (Negatif), atau **Neutral** (Netral).\n\n"
        "Model ini dilatih menggunakan **PyCaret** dengan **TF-IDF vectorization** dan "
        "custom preprocessing untuk menangani slang & leetspeak Bahasa Indonesia."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

# ══════════════════════════════════════════════
# 🚀 LAUNCH APPLICATION
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Gradio App for Sentiment Analysis")
    print("=" * 60)
    print("📱 Local URL: http://127.0.0.1:7860")
    print("🌐 Share link will be available after launch")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
