from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
RAW_CSV = PROJECT_DIR / "data" / "raw" / "sentimentdataset.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
README_PATH = PROJECT_DIR / "README.md"
OUTPUT_CSV = PROCESSED_DIR / "clean_data.csv"
OUTPUT_NOTEBOOK = NOTEBOOK_DIR / "01_eda_preprocessing.ipynb"

# Folder tambahan mengikuti struktur target repository
MODELS_DIR = PROJECT_DIR / "models"
APP_DIR = PROJECT_DIR / "app"


# ============================================================
# KONSTANTA & MAPPING
# ============================================================
POSITIVE_LABELS = {
    "Acceptance",
    "Accomplishment",
    "Admiration",
    "Adoration",
    "Adrenaline",
    "Adventure",
    "Affection",
    "Amazement",
    "Amusement",
    "Anticipation",
    "Appreciation",
    "Arousal",
    "ArtisticBurst",
    "Awe",
    "Blessed",
    "Breakthrough",
    "Calmness",
    "Captivation",
    "Celebration",
    "Celestial Wonder",
    "Challenge",
    "Charm",
    "Colorful",
    "Compassion",
    "Compassionate",
    "Confidence",
    "Confident",
    "Connection",
    "Contentment",
    "Coziness",
    "Creative Inspiration",
    "Creativity",
    "Culinary Adventure",
    "CulinaryOdyssey",
    "Dazzle",
    "Determination",
    "DreamChaser",
    "Ecstasy",
    "Elation",
    "Elegance",
    "Empathetic",
    "Empowerment",
    "Enchantment",
    "Energy",
    "Engagement",
    "Enjoyment",
    "Enthusiasm",
    "Envisioning History",
    "Euphoria",
    "Excitement",
    "Exploration",
    "FestiveJoy",
    "Free-spirited",
    "Freedom",
    "Friendship",
    "Fulfillment",
    "Grandeur",
    "Grateful",
    "Gratitude",
    "Happiness",
    "Happy",
    "Harmony",
    "Heartwarming",
    "Hope",
    "Hopeful",
    "Hypnotic",
    "Iconic",
    "Imagination",
    "Immersion",
    "Inspired",
    "Inspiration",
    "Intrigue",
    "Journey",
    "Joy",
    "Joy in Baking",
    "JoyfulReunion",
    "Kind",
    "Kindness",
    "Love",
    "Marvel",
    "Melodic",
    "Mesmerizing",
    "Mindfulness",
    "Mischievous",
    "Motivation",
    "Nature's Beauty",
    "Ocean's Freedom",
    "Optimism",
    "Overjoyed",
    "Playful",
    "PlayfulJoy",
    "Positive",
    "Positivity",
    "Pride",
    "Proud",
    "Radiance",
    "Rejuvenation",
    "Relief",
    "Renewed Effort",
    "Resilience",
    "Reverence",
    "Romance",
    "Runway Creativity",
    "Satisfaction",
    "Serenity",
    "Solace",
    "Spark",
    "Success",
    "Sympathy",
    "Tenderness",
    "Thrill",
    "Thrilling Journey",
    "Touched",
    "Tranquility",
    "Triumph",
    "Vibrancy",
    "Whimsy",
    "Winter Magic",
    "Wonder",
    "Wonderment",
    "Zest",
}

NEGATIVE_LABELS = {
    "Anger",
    "Anxiety",
    "Apprehensive",
    "Bad",
    "Betrayal",
    "Bitter",
    "Bitterness",
    "Boredom",
    "Darkness",
    "Desolation",
    "Despair",
    "Desperation",
    "Devastated",
    "Disappointed",
    "Disappointment",
    "Disgust",
    "Dismissive",
    "Embarrassed",
    "EmotionalStorm",
    "Envious",
    "Envy",
    "Exhaustion",
    "Fear",
    "Fearful",
    "Frustrated",
    "Frustration",
    "Grief",
    "Hate",
    "Heartache",
    "Heartbreak",
    "Helplessness",
    "Intimidation",
    "Isolation",
    "Jealous",
    "Jealousy",
    "Loneliness",
    "Loss",
    "LostLove",
    "Melancholy",
    "Miscalculation",
    "Negative",
    "Numbness",
    "Obstacle",
    "Overwhelmed",
    "Pressure",
    "Regret",
    "Resentment",
    "Ruins",
    "Sad",
    "Sadness",
    "Shame",
    "Solitude",
    "Sorrow",
    "Suffering",
    "Suspense",
    "Whispers of the Past",
    "Yearning",
}

CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "ain't": "is not",
    "let's": "let us",
    "isn't": "is not",
    "it's": "it is",
    "i've": "i have",
    "doesn't": "does not",
    "didn't": "did not",
    "don't": "do not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
}

NOTEBOOK_EXECUTION_INSTRUCTIONS = """# Cara menjalankan notebook
# 1. Pastikan file mentah tersedia di: ../data/raw/sentimentdataset.csv
# 2. Jalankan notebook ini dari folder notebooks/
# 3. Hasil preprocessing akan disimpan ke: ../data/processed/clean_data.csv
"""


# ============================================================
# UTILITAS PREPROCESSING
# ============================================================
def ensure_directories() -> None:
    """Buat seluruh direktori target jika belum tersedia."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    APP_DIR.mkdir(parents=True, exist_ok=True)


def expand_contractions(text: str) -> str:
    """Ekspansi kontraksi bahasa Inggris berbasis token sederhana."""
    words = text.split()
    expanded = [CONTRACTION_MAP.get(word, word) for word in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """
    Pipeline pembersihan teks bergaya mct-nlp:
    lowercase -> hapus HTML -> hapus URL -> ekspansi kontraksi
    -> ubah hashtag menjadi token biasa -> hapus karakter non-alfabet
    -> rapikan spasi.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = expand_contractions(text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def zscore(series: pd.Series) -> pd.Series:
    """Hitung z-score tanpa dependensi eksternal."""
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index, dtype="float64")
    return ((series - series.mean()) / std).astype("float64")


def dataframe_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Konversi DataFrame ke tabel markdown tanpa dependensi eksternal."""
    table = df.copy()

    if index:
        table = table.reset_index()
        if "index" in table.columns:
            table = table.rename(columns={"index": ""})

    headers = [str(col) for col in table.columns]
    separator = ["---"] * len(headers)

    rows = []
    for _, row in table.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                cell = ""
            else:
                cell = str(value)
            cell = cell.replace("\n", " ").replace("|", "\\|")
            values.append(cell)
        rows.append("| " + " | ".join(values) + " |")

    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_lines.extend(rows)
    return "\n".join(markdown_lines)


def map_sentiment_group(label: str) -> str:
    """Kelompokkan label sentimen granular menjadi 3 kelas."""
    if label in POSITIVE_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    return "neutral"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Lakukan pembersihan, rekayasa fitur, dan standardisasi kolom."""
    df = df.copy()

    # Standardisasi nama kolom
    df.columns = [col.strip() for col in df.columns]

    rename_map = {
        "Unnamed: 0.1": "record_id",
        "Unnamed: 0": "source_index",
        "Text": "text",
        "Sentiment": "sentiment",
        "Timestamp": "timestamp",
        "User": "user",
        "Platform": "platform",
        "Hashtags": "hashtags",
        "Retweets": "retweets",
        "Likes": "likes",
        "Country": "country",
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Hour": "hour",
    }
    df = df.rename(columns=rename_map)

    # Pembersihan dasar kolom teks/kategori
    object_columns = [
        "text",
        "sentiment",
        "user",
        "platform",
        "hashtags",
        "country",
        "timestamp",
    ]
    for col in object_columns:
        df[col] = df[col].astype(str).str.strip()

    # Parsing numerik dan waktu
    numeric_columns = [
        "record_id",
        "source_index",
        "retweets",
        "likes",
        "year",
        "month",
        "day",
        "hour",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce"
    )

    # Drop missing pada kolom utama
    df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    # Feature engineering
    df["sentiment"] = df["sentiment"].str.strip()
    df["sentiment_group"] = df["sentiment"].apply(map_sentiment_group)
    df["cleaned_text"] = df["text"].apply(clean_text)

    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    df["text_length_chars"] = df["text"].str.len()
    df["text_length_words"] = df["text"].str.split().str.len()
    df["cleaned_text_length_chars"] = df["cleaned_text"].str.len()
    df["cleaned_text_length_words"] = df["cleaned_text"].str.split().str.len()
    df["has_hashtag"] = df["hashtags"].str.contains(r"#", regex=True).astype(int)
    df["hashtag_count"] = df["hashtags"].str.count(r"#")
    df["engagement_total"] = df["retweets"].fillna(0) + df["likes"].fillna(0)

    # Scaling numerik
    scale_targets = [
        "retweets",
        "likes",
        "engagement_total",
        "text_length_chars",
        "text_length_words",
        "cleaned_text_length_chars",
        "cleaned_text_length_words",
        "hashtag_count",
    ]
    for col in scale_targets:
        df[f"{col}_scaled"] = zscore(df[col]).round(6)

    # Rapikan urutan kolom
    ordered_columns = [
        "record_id",
        "source_index",
        "timestamp",
        "timestamp_dt",
        "year",
        "month",
        "day",
        "hour",
        "user",
        "platform",
        "country",
        "hashtags",
        "has_hashtag",
        "hashtag_count",
        "text",
        "cleaned_text",
        "text_length_chars",
        "text_length_words",
        "cleaned_text_length_chars",
        "cleaned_text_length_words",
        "sentiment",
        "sentiment_group",
        "retweets",
        "likes",
        "engagement_total",
        "retweets_scaled",
        "likes_scaled",
        "engagement_total_scaled",
        "text_length_chars_scaled",
        "text_length_words_scaled",
        "cleaned_text_length_chars_scaled",
        "cleaned_text_length_words_scaled",
        "hashtag_count_scaled",
    ]
    return df[ordered_columns]


# ============================================================
# RINGKASAN EDA
# ============================================================
def build_analysis_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Bangun ringkasan statistik untuk README dan notebook."""
    sentiment_counts = df_clean["sentiment"].value_counts().head(15).to_dict()
    polarity_counts = df_clean["sentiment_group"].value_counts().to_dict()
    platform_counts = df_clean["platform"].value_counts().to_dict()
    country_counts = df_clean["country"].value_counts().head(10).to_dict()

    numeric_corr = (
        df_clean[
            [
                "retweets",
                "likes",
                "engagement_total",
                "text_length_chars",
                "text_length_words",
                "cleaned_text_length_chars",
                "cleaned_text_length_words",
                "hour",
                "day",
                "month",
                "year",
            ]
        ]
        .corr(numeric_only=True)
        .round(3)
    )

    duplicates_text = int(df_clean["text"].duplicated().sum())
    emoji_rows = int(df_clean["text"].str.contains(r"[^\x00-\x7F]", regex=True).sum())
    hashtag_rows = int(df_clean["text"].str.contains(r"#\w+", regex=True).sum())

    summary = {
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "n_columns_raw": int(df_raw.shape[1]),
        "n_columns_clean": int(df_clean.shape[1]),
        "missing_total_raw": int(df_raw.isna().sum().sum()),
        "unique_sentiments": int(df_clean["sentiment"].nunique()),
        "duplicates_text": duplicates_text,
        "emoji_rows": emoji_rows,
        "hashtag_in_text_rows": hashtag_rows,
        "sentiment_counts_top15": sentiment_counts,
        "polarity_counts": polarity_counts,
        "platform_counts": platform_counts,
        "country_counts_top10": country_counts,
        "correlation_markdown": dataframe_to_markdown(numeric_corr),
        "sample_before_after": dataframe_to_markdown(
            df_clean[["text", "cleaned_text", "sentiment", "sentiment_group"]].sample(
                n=min(8, len(df_clean)), random_state=42
            ),
            index=False,
        ),
    }
    return summary


# ============================================================
# GENERATOR README
# ============================================================
def generate_readme(summary: dict) -> str:
    """Bangun README berbahasa Indonesia untuk Person 1."""
    sentiment_lines = "\n".join(
        [
            f"- **{label}**: {count}"
            for label, count in summary["sentiment_counts_top15"].items()
        ]
    )
    polarity_lines = "\n".join(
        [
            f"- **{label}**: {count}"
            for label, count in summary["polarity_counts"].items()
        ]
    )
    platform_lines = "\n".join(
        [
            f"- **{label}**: {count}"
            for label, count in summary["platform_counts"].items()
        ]
    )
    country_lines = "\n".join(
        [
            f"- **{label}**: {count}"
            for label, count in summary["country_counts_top10"].items()
        ]
    )

    return f"""# Project ML - Sentiment Analysis

## Deskripsi
Repositori ini disiapkan untuk project **Sentiment Analysis end-to-end** dengan:
- gaya preprocessing dan dokumentasi yang mengacu pada folder `mct-nlp`
- struktur deployment yang mengacu pada folder `deteksi-toksisitas-chat`

Dokumentasi pada file ini difokuskan untuk **Person 1 - Data Analyst (Pre-processing Specialist)**.

## Struktur Folder
```text
project-ml/
├── data/
│   ├── raw/
│   │   └── sentimentdataset.csv
│   └── processed/
│       └── clean_data.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── models/
├── app/
└── README.md
```

## Ringkasan Observasi Awal
### Dataset utama
- Jumlah baris mentah: **{summary["raw_rows"]}**
- Jumlah kolom mentah: **{summary["n_columns_raw"]}**
- Total missing value pada data mentah: **{summary["missing_total_raw"]}**
- Jumlah baris setelah preprocessing: **{summary["clean_rows"]}**
- Jumlah kolom setelah preprocessing: **{summary["n_columns_clean"]}**
- Jumlah label sentimen unik: **{summary["unique_sentiments"]}**
- Jumlah duplikasi pada kolom teks: **{summary["duplicates_text"]}**
- Baris yang mengandung emoji/non-ASCII pada teks: **{summary["emoji_rows"]}**
- Baris yang mengandung hashtag langsung di kolom teks: **{summary["hashtag_in_text_rows"]}**

### Top 15 distribusi label sentimen asli
{sentiment_lines}

### Distribusi sentimen hasil pengelompokan 3 kelas
{polarity_lines}

### Distribusi platform
{platform_lines}

### Top 10 negara
{country_lines}

## Standar Preprocessing yang Diadopsi dari `mct-nlp`
Berdasarkan observasi pada modul referensi, pola preprocessing yang diadopsi adalah:
1. mempertahankan kolom teks asli
2. membuat kolom baru bernama `cleaned_text`
3. menggunakan pipeline pembersihan yang eksplisit dan mudah dibaca
4. menghindari preprocessing berlebihan
5. menambahkan contoh before vs after untuk dokumentasi notebook

### Pipeline pembersihan teks
Urutan preprocessing yang digunakan:
1. `lowercase`
2. hapus tag HTML
3. hapus URL
4. ekspansi kontraksi bahasa Inggris
5. ubah hashtag di dalam teks menjadi token biasa
6. hapus mention
7. hapus karakter non-alfabet
8. rapikan whitespace

## Rekayasa Fitur Tambahan
Selain `cleaned_text`, file `clean_data.csv` juga menyimpan fitur tambahan:
- `sentiment_group` sebagai versi agregasi 3 kelas: `positive`, `negative`, `neutral`
- `text_length_chars`
- `text_length_words`
- `cleaned_text_length_chars`
- `cleaned_text_length_words`
- `has_hashtag`
- `hashtag_count`
- `engagement_total`
- kolom hasil scaling z-score untuk fitur numerik utama

## Catatan Pengelompokan Sentimen
Dataset asli memiliki ratusan label emosi granular. Agar lebih mudah digunakan pada tahap modeling, dibuat kolom:
- `sentiment` -> label asli
- `sentiment_group` -> hasil agregasi menjadi 3 kelas

Pendekatan ini dipilih agar:
- analisis EDA tetap mempertahankan label asli
- modeling tahap awal lebih realistis untuk klasifikasi sentimen umum

## Korelasi Fitur Numerik
Berikut korelasi antar fitur numerik utama:

{summary["correlation_markdown"]}

## Contoh Sebelum dan Sesudah Preprocessing
{summary["sample_before_after"]}

## File yang Dihasilkan
- `data/processed/clean_data.csv`
- `notebooks/01_eda_preprocessing.ipynb`

## Cara Menjalankan Generator
Jalankan perintah berikut dari folder `project-ml`:

```bash
python generate_person1_assets.py
```

## Tugas Person 1 yang Sudah Dicakup
- observasi referensi `mct-nlp`
- observasi dataset `sentimentdataset.csv`
- EDA awal
- preprocessing sesuai standar referensi
- penyimpanan data bersih ke `data/processed/clean_data.csv`
- dokumentasi notebook dan README dalam bahasa Indonesia

## Catatan Lanjutan
Tahap berikutnya yang dapat dikerjakan:
- pembuatan `02_modeling_pycaret.ipynb`
- training model berbasis `cleaned_text` atau kombinasi fitur teks + metadata
- penyusunan app demo mengikuti pola deployment Hugging Face Space
"""


# ============================================================
# GENERATOR NOTEBOOK
# ============================================================
def to_source(text: str) -> list[str]:
    """Ubah string multiline menjadi format source notebook."""
    return [line + "\n" for line in text.strip("\n").split("\n")]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


def generate_notebook(summary: dict) -> dict:
    """Bangun notebook EDA + preprocessing dengan gaya workshop."""
    md_intro = f"""
# 📘 Notebook 01 - EDA dan Preprocessing Sentiment Analysis

Notebook ini disusun untuk kebutuhan **Person 1 - Data Analyst (Pre-processing Specialist)**.

## Tujuan Notebook
1. memuat dataset mentah
2. melakukan EDA awal
3. menerapkan preprocessing teks dengan gaya `mct-nlp`
4. membuat fitur tambahan untuk modeling
5. menyimpan data bersih ke folder `data/processed`

## Ringkasan Temuan Awal
- Jumlah baris data mentah: **{summary["raw_rows"]}**
- Jumlah label sentimen unik: **{summary["unique_sentiments"]}**
- Tidak ditemukan missing value pada data mentah
- Ditemukan **{summary["duplicates_text"]}** duplikasi pada kolom teks
- Distribusi agregasi sentimen 3 kelas:
  - positive: **{summary["polarity_counts"].get("positive", 0)}**
  - negative: **{summary["polarity_counts"].get("negative", 0)}**
  - neutral: **{summary["polarity_counts"].get("neutral", 0)}**
"""

    md_reference = """
## 🔎 Ringkasan Referensi `mct-nlp`

Pola utama yang diadopsi dari referensi:
- fungsi preprocessing dibuat sederhana dan modular
- hasil pembersihan disimpan pada kolom `cleaned_text`
- preprocessing dilakukan secara rule-based
- dokumentasi dibuat edukatif dan mudah dipresentasikan

Untuk dataset ini, pipeline dibangun dengan pendekatan:
`lowercase -> hapus HTML -> hapus URL -> ekspansi kontraksi -> normalisasi hashtag -> hapus mention -> hapus non-alfabet -> rapikan spasi`
"""

    code_imports = f"""
{NOTEBOOK_EXECUTION_INSTRUCTIONS}

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
pd.set_option("display.max_colwidth", 120)

BASE_DIR = Path.cwd().resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "sentimentdataset.csv"
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "clean_data.csv"

print("📂 File mentah:", RAW_CSV)
print("💾 File output:", PROCESSED_CSV)
"""

    code_helpers = """
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "ain't": "is not",
    "let's": "let us",
    "isn't": "is not",
    "it's": "it is",
    "i've": "i have",
    "doesn't": "does not",
    "didn't": "did not",
    "don't": "do not",
    "wasn't": "was not",
    "weren't": "were not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
}

POSITIVE_LABELS = {
    "Acceptance","Accomplishment","Admiration","Adoration","Adrenaline","Adventure","Affection","Amazement","Amusement",
    "Anticipation","Appreciation","Arousal","ArtisticBurst","Awe","Blessed","Breakthrough","Calmness","Captivation",
    "Celebration","Celestial Wonder","Challenge","Charm","Colorful","Compassion","Compassionate","Confidence","Confident",
    "Connection","Contentment","Coziness","Creative Inspiration","Creativity","Culinary Adventure","CulinaryOdyssey",
    "Dazzle","Determination","DreamChaser","Ecstasy","Elation","Elegance","Empathetic","Empowerment","Enchantment",
    "Energy","Engagement","Enjoyment","Enthusiasm","Envisioning History","Euphoria","Excitement","Exploration",
    "FestiveJoy","Free-spirited","Freedom","Friendship","Fulfillment","Grandeur","Grateful","Gratitude","Happiness",
    "Happy","Harmony","Heartwarming","Hope","Hopeful","Hypnotic","Iconic","Imagination","Immersion","Inspired",
    "Inspiration","Intrigue","Journey","Joy","Joy in Baking","JoyfulReunion","Kind","Kindness","Love","Marvel",
    "Melodic","Mesmerizing","Mindfulness","Mischievous","Motivation","Nature's Beauty","Ocean's Freedom","Optimism",
    "Overjoyed","Playful","PlayfulJoy","Positive","Positivity","Pride","Proud","Radiance","Rejuvenation","Relief",
    "Renewed Effort","Resilience","Reverence","Romance","Runway Creativity","Satisfaction","Serenity","Solace","Spark",
    "Success","Sympathy","Tenderness","Thrill","Thrilling Journey","Touched","Tranquility","Triumph","Vibrancy",
    "Whimsy","Winter Magic","Wonder","Wonderment","Zest"
}

NEGATIVE_LABELS = {
    "Anger","Anxiety","Apprehensive","Bad","Betrayal","Bitter","Bitterness","Boredom","Darkness","Desolation","Despair",
    "Desperation","Devastated","Disappointed","Disappointment","Disgust","Dismissive","Embarrassed","EmotionalStorm",
    "Envious","Envy","Exhaustion","Fear","Fearful","Frustrated","Frustration","Grief","Hate","Heartache","Heartbreak",
    "Helplessness","Intimidation","Isolation","Jealous","Jealousy","Loneliness","Loss","LostLove","Melancholy",
    "Miscalculation","Negative","Numbness","Obstacle","Overwhelmed","Pressure","Regret","Resentment","Ruins","Sad",
    "Sadness","Shame","Solitude","Sorrow","Suffering","Suspense","Whispers of the Past","Yearning"
}

def expand_contractions(text: str) -> str:
    words = text.split()
    expanded = [CONTRACTION_MAP.get(word, word) for word in words]
    return " ".join(expanded)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = expand_contractions(text)
    text = re.sub(r"#(\\w+)", r" \\1 ", text)
    text = re.sub(r"@\\w+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def zscore(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index, dtype="float64")
    return ((series - series.mean()) / std).astype("float64")

def map_sentiment_group(label: str) -> str:
    if label in POSITIVE_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    return "neutral"
"""

    code_load = """
print("📥 Membaca dataset mentah...")
df_raw = pd.read_csv(RAW_CSV, sep=";")
print("Shape data mentah:", df_raw.shape)
display(df_raw.head())
"""

    code_eda = """
print("🔎 Informasi dataset")
display(pd.DataFrame({
    "kolom": df_raw.columns,
    "dtype": df_raw.dtypes.astype(str).values
}))

print("🧼 Missing value tiap kolom")
display(df_raw.isna().sum().to_frame("missing_count"))

print("🏷️ Top 15 label sentimen")
df_sentiment = (
    df_raw["Sentiment"]
    .astype(str)
    .str.strip()
    .value_counts()
    .head(15)
    .reset_index()
)
df_sentiment.columns = ["sentiment", "jumlah"]
display(df_sentiment)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_sentiment, x="jumlah", y="sentiment", palette="viridis")
plt.title("Top 15 Distribusi Sentimen Asli")
plt.xlabel("Jumlah")
plt.ylabel("Sentimen")
plt.tight_layout()
plt.show()

print("📱 Distribusi platform")
df_platform = (
    df_raw["Platform"]
    .astype(str)
    .str.strip()
    .value_counts()
    .reset_index()
)
df_platform.columns = ["platform", "jumlah"]
display(df_platform)

plt.figure(figsize=(8, 4))
sns.barplot(data=df_platform, x="platform", y="jumlah", palette="magma")
plt.title("Distribusi Platform")
plt.xlabel("Platform")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()
"""

    code_preprocess = """
print("🧹 Memulai preprocessing...")

df = df_raw.copy()
df.columns = [col.strip() for col in df.columns]

rename_map = {
    "Unnamed: 0.1": "record_id",
    "Unnamed: 0": "source_index",
    "Text": "text",
    "Sentiment": "sentiment",
    "Timestamp": "timestamp",
    "User": "user",
    "Platform": "platform",
    "Hashtags": "hashtags",
    "Retweets": "retweets",
    "Likes": "likes",
    "Country": "country",
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "Hour": "hour",
}
df = df.rename(columns=rename_map)

object_columns = ["text", "sentiment", "user", "platform", "hashtags", "country", "timestamp"]
for col in object_columns:
    df[col] = df[col].astype(str).str.strip()

numeric_columns = ["record_id", "source_index", "retweets", "likes", "year", "month", "day", "hour"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")

df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)
df["sentiment_group"] = df["sentiment"].apply(map_sentiment_group)
df["cleaned_text"] = df["text"].apply(clean_text)
df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

df["text_length_chars"] = df["text"].str.len()
df["text_length_words"] = df["text"].str.split().str.len()
df["cleaned_text_length_chars"] = df["cleaned_text"].str.len()
df["cleaned_text_length_words"] = df["cleaned_text"].str.split().str.len()
df["has_hashtag"] = df["hashtags"].str.contains(r"#", regex=True).astype(int)
df["hashtag_count"] = df["hashtags"].str.count(r"#")
df["engagement_total"] = df["retweets"].fillna(0) + df["likes"].fillna(0)

scale_targets = [
    "retweets",
    "likes",
    "engagement_total",
    "text_length_chars",
    "text_length_words",
    "cleaned_text_length_chars",
    "cleaned_text_length_words",
    "hashtag_count",
]

for col in scale_targets:
    df[f"{col}_scaled"] = zscore(df[col]).round(6)

ordered_columns = [
    "record_id","source_index","timestamp","timestamp_dt","year","month","day","hour",
    "user","platform","country","hashtags","has_hashtag","hashtag_count",
    "text","cleaned_text","text_length_chars","text_length_words",
    "cleaned_text_length_chars","cleaned_text_length_words",
    "sentiment","sentiment_group","retweets","likes","engagement_total",
    "retweets_scaled","likes_scaled","engagement_total_scaled",
    "text_length_chars_scaled","text_length_words_scaled",
    "cleaned_text_length_chars_scaled","cleaned_text_length_words_scaled",
    "hashtag_count_scaled"
]

df_clean = df[ordered_columns].copy()

print("✅ Preprocessing selesai")
print("Shape data bersih:", df_clean.shape)
display(df_clean.head())
"""

    code_examples = """
print("📋 Contoh sebelum dan sesudah preprocessing")
examples = (
    df_clean[["text", "cleaned_text", "sentiment", "sentiment_group"]]
    .sample(n=min(10, len(df_clean)), random_state=42)
    .reset_index(drop=True)
)
display(examples)
"""

    code_polarity = """
print("📊 Distribusi sentimen 3 kelas")
df_polarity = df_clean["sentiment_group"].value_counts().reset_index()
df_polarity.columns = ["sentiment_group", "jumlah"]
display(df_polarity)

plt.figure(figsize=(6, 4))
sns.barplot(data=df_polarity, x="sentiment_group", y="jumlah", palette="Set2")
plt.title("Distribusi Sentiment Group")
plt.xlabel("Kelompok Sentimen")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()
"""

    code_corr = """
print("🔗 Korelasi fitur numerik utama")
corr_cols = [
    "retweets", "likes", "engagement_total",
    "text_length_chars", "text_length_words",
    "cleaned_text_length_chars", "cleaned_text_length_words",
    "hour", "day", "month", "year"
]
corr_matrix = df_clean[corr_cols].corr(numeric_only=True)
display(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.show()
"""

    code_save = """
print("💾 Menyimpan clean_data.csv ...")
PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(PROCESSED_CSV, index=False)
print("File berhasil disimpan:", PROCESSED_CSV)
"""

    md_closing = """
## ✅ Kesimpulan
Pada notebook ini telah dilakukan:
- pembacaan dataset mentah
- EDA awal
- pembersihan teks dengan standar yang mengikuti gaya `mct-nlp`
- pembentukan fitur numerik tambahan
- pengelompokan label granular menjadi `sentiment_group`
- penyimpanan hasil akhir ke `data/processed/clean_data.csv`

Notebook ini siap menjadi dasar untuk tahap modeling pada notebook berikutnya.
"""

    return {
        "cells": [
            markdown_cell(md_intro),
            markdown_cell(md_reference),
            code_cell(code_imports),
            code_cell(code_helpers),
            code_cell(code_load),
            code_cell(code_eda),
            code_cell(code_preprocess),
            code_cell(code_examples),
            code_cell(code_polarity),
            code_cell(code_corr),
            code_cell(code_save),
            markdown_cell(md_closing),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ============================================================
# EKSEKUSI UTAMA
# ============================================================
def main() -> None:
    print("🚀 Memulai generator aset Person 1...")
    ensure_directories()

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"File dataset mentah tidak ditemukan di: {RAW_CSV}")

    print(f"📂 Membaca dataset: {RAW_CSV}")
    df_raw = pd.read_csv(RAW_CSV, sep=";")

    print("🧹 Melakukan preprocessing data...")
    df_clean = preprocess_dataframe(df_raw)

    print(f"💾 Menyimpan data bersih ke: {OUTPUT_CSV}")
    df_clean.to_csv(OUTPUT_CSV, index=False)

    print("📝 Menyusun ringkasan analisis...")
    summary = build_analysis_summary(df_raw, df_clean)

    print(f"📘 Menulis README ke: {README_PATH}")
    README_PATH.write_text(generate_readme(summary), encoding="utf-8")

    print(f"📓 Menulis notebook ke: {OUTPUT_NOTEBOOK}")
    notebook_content = generate_notebook(summary)
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook_content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Selesai! Aset Person 1 berhasil dibuat.")
    print(f"- CSV bersih  : {OUTPUT_CSV}")
    print(f"- Notebook    : {OUTPUT_NOTEBOOK}")
    print(f"- README      : {README_PATH}")


if __name__ == "__main__":
    main()
