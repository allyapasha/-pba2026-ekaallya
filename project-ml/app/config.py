"""
config.py — Konfigurasi untuk Sentiment Analysis Gradio App
============================================================
Berisi path configuration, preprocessing mappings, dan constants
yang digunakan oleh app.py, mengikuti standar mct-nlp.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 📁 PATH CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model path
MODEL_PATH = MODELS_DIR / "sentiment_pycaret_best"
MODEL_PKL_PATH = str(MODEL_PATH)  # For PyCaret load_model()

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🔤 LEETSPEAK MAPPING
# ──────────────────────────────────────────────
# Gamer & internet slang Indonesia menggunakan angka/simbol
# sebagai pengganti huruf untuk menghindari filter kata kasar
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

# ──────────────────────────────────────────────
# 💬 KAMUS SLANG BAHASA INDONESIA
# ──────────────────────────────────────────────
# Singkatan & slang yang umum di chat Indonesia
# Digunakan untuk normalisasi teks sebelum TF-IDF
SLANG_DICT = {
    # --- Kata kasar / toxic (untuk konteks umum) ---
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
    # --- Slang umum ---
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
    # --- Gaming terms ---
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
    # --- Sentiment-related ---
    "bagus": "bagus",
    "buruk": "buruk",
    "jelek": "jelek",
    "suka": "suka",
    "benci": "benci",
    "sayang": "sayang",
    "senang": "senang",
    "sedih": "sedih",
    "marah": "marah",
    "mantap": "mantap",
    "oke": "ok",
    "ok": "ok",
}

# ──────────────────────────────────────────────
# 🎯 GRADIO INTERFACE CONFIGURATION
# ──────────────────────────────────────────────

# Title & description
APP_TITLE = "😊 Sentiment Analysis - Indonesian Text"

APP_DESCRIPTION = """
Analisis sentimen teks berbahasa Indonesia menggunakan Machine Learning.

Model ini mengklasifikasikan sentimen teks ke dalam 3 kategori:
- 🟢 **Positive** - Teks dengan sentimen positif/baik
- 🔴 **Negative** - Teks dengan sentimen negatif/buruk
- 🟡 **Neutral** - Teks dengan sentimen netral/biasa saja

Dilatih menggunakan **PyCaret** dengan **TF-IDF vectorization** dan
custom preprocessing untuk menangani slang & leetspeak Bahasa Indonesia.
"""

# Example inputs untuk Gradio interface
GRADIO_EXAMPLES = [
    ["Saya sangat senang dengan produk ini! Kualitasnya luar biasa!"],
    ["Terrible service, never coming back here!"],
    ["Harga cukup mahal tetapi kualitasnya standar."],
    ["This is the best day of my life!"],
    ["Layanan pelanggan mereka sangat buruk dan lamban."],
    ["Produk ini biasa saja, tidak istimewa."],
    ["Absolutely fantastic experience, highly recommend!"],
    ["Tidak puas dengan hasil pembelian saya."],
    ["Amazing quality dan harga yang reasonable!"],
    ["Kecewa sekali dengan pelayanan mereka."],
]

# ──────────────────────────────────────────────
# ⚙️ PYCARET CONFIGURATION
# ──────────────────────────────────────────────

# Model training parameters (used in notebook)
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
SESSION_ID = 42  # Random seed untuk reproducibility
N_FOLDS = 5
FOLD_STRATEGY = "stratified"  # Important untuk imbalanced data
OPTIMIZE_METRIC = "Accuracy"  # Can also use: F1, AUC, Precision, Recall

# Target column name
TARGET_COL = "sentiment_group"

# Text column name (after preprocessing)
TEXT_COL = "cleaned_text"

# Possible target classes
SENTIMENT_CLASSES = ["positive", "negative", "neutral"]

# ──────────────────────────────────────────────
# 🎨 GRADIO THEME & STYLING
# ──────────────────────────────────────────────

GRADIO_THEME = "soft"  # Options: default, soft, huggingface, glass, monochrome
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
GRADIO_DEBUG = False

# ──────────────────────────────────────────────
# 📊 PREPROCESSING CONSTANTS
# ──────────────────────────────────────────────

# URL pattern untuk removal
URL_PATTERN = r"https?://\S+|www\.\S+"

# Mention pattern (@username)
MENTION_PATTERN = r"@\w+"

# Hashtag pattern (#tag)
HASHTAG_PATTERN = r"#\w+"

# Non-alphanumeric pattern (keep spaces)
NON_ALNUM_PATTERN = r"[^a-z\s]"

# Whitespace pattern
WHITESPACE_PATTERN = r"\s+"

# ──────────────────────────────────────────────
# 🔄 PREPROCESSING PIPELINE SETTINGS
# ──────────────────────────────────────────────

# Preprocessing steps (in order)
PREPROCESSING_STEPS = [
    "lowercase",  # Convert to lowercase
    "remove_urls",  # Remove http/https/www URLs
    "remove_mentions",  # Remove @username mentions
    "normalize_leetspeak",  # 4nj1n9 → anjing
    "expand_slang",  # gw → gue, lu → lo
    "remove_non_alnum",  # Remove punctuation & special chars
    "normalize_whitespace",  # Clean up spaces
]

# ──────────────────────────────────────────────
# ✅ VALIDATION & ERROR HANDLING
# ──────────────────────────────────────────────

# Minimum text length after cleaning (in characters)
MIN_TEXT_LENGTH = 1

# Maximum text length (in characters) - for display/processing
MAX_TEXT_LENGTH = 1000

# Empty text response
EMPTY_TEXT_RESPONSE = {"Error": 1.0}

# Error prefix untuk error messages
ERROR_PREFIX = "⚠️ Error: "

# ──────────────────────────────────────────────
# 📱 UI/UX CONFIGURATION
# ──────────────────────────────────────────────

# Input textbox settings
INPUT_PLACEHOLDER = (
    "Ketik teks untuk analisis sentimen... (contoh: 'Saya sangat menyukai produk ini!')"
)

INPUT_LABEL = "💬 Masukkan Teks"
INPUT_LINES = 5

# Output label settings
OUTPUT_LABEL = "🎯 Hasil Prediksi Sentimen"
OUTPUT_NUM_TOP_CLASSES = 3

# Button label
BUTTON_LABEL = "🔍 Analyze Sentiment"

# ──────────────────────────────────────────────
# 🔧 LOGGING & DEBUG CONFIGURATION
# ──────────────────────────────────────────────

# Enable verbose logging
VERBOSE = False

# Log file path (optional)
LOG_FILE = None  # Set to APP_DIR / "logs" / "app.log" to enable

# ──────────────────────────────────────────────
# 📋 INFO & METADATA
# ──────────────────────────────────────────────

APP_VERSION = "1.0.0"
APP_AUTHOR = "Person 2 - ML Engineer"
APP_DATE = "2024"
APP_INSTITUTION = "Institut Teknologi Sumatera"
APP_COURSE = "SD25-32202 - Pemrosesan Bahasa Alami"

# Project reference folders
PROJECT_REFERENCE = {
    "preprocessing_standard": "mct-nlp",
    "deployment_template": "deteksi-toksisitas-chat",
}

# ──────────────────────────────────────────────
# 🚀 DEPLOYMENT SETTINGS
# ──────────────────────────────────────────────

# Hugging Face Spaces configuration
HF_SPACE_NAME = "sentiment-analysis-indonesian"
HF_SPACE_CONFIG = {
    "title": "Sentiment Analysis - Indonesian Text",
    "emoji": "😊",
    "colorFrom": "blue",
    "colorTo": "purple",
    "sdk": "gradio",
    "sdk_version": "5.20.1",
    "python_version": "3.10",
    "app_file": "app.py",
    "pinned": False,
    "license": "mit",
}

# ──────────────────────────────────────────────
# 🔍 UTILITY FUNCTIONS (OPTIONAL)
# ──────────────────────────────────────────────


def get_config_summary() -> str:
    """Return a summary of current configuration."""
    summary = f"""
╔════════════════════════════════════════════╗
║          APP CONFIGURATION SUMMARY          ║
╠════════════════════════════════════════════╣
║
║ Application
║   Title: {APP_TITLE}
║   Version: {APP_VERSION}
║   Author: {APP_AUTHOR}
║
║ Directories
║   Base: {BASE_DIR}
║   App: {APP_DIR}
║   Models: {MODELS_DIR}
║
║ Model
║   Path: {MODEL_PKL_PATH}
║   Classes: {", ".join(SENTIMENT_CLASSES)}
║
║ Server
║   Host: {GRADIO_SERVER_NAME}
║   Port: {GRADIO_SERVER_PORT}
║   Theme: {GRADIO_THEME}
║
║ Preprocessing
║   Steps: {len(PREPROCESSING_STEPS)}
║   Slang dict size: {len(SLANG_DICT)} entries
║   Leetspeak map size: {len(LEETSPEAK_MAP)} entries
║
╚════════════════════════════════════════════╝
"""
    return summary


if __name__ == "__main__":
    # Display configuration when run directly
    print(get_config_summary())
    print("\n✅ Configuration loaded successfully!")
