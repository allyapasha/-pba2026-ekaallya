from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TEXT_COLUMN = "Text"
LABEL_COLUMN = "Sentiment"
TARGET_COLUMN = "sentiment_group"
NUMERIC_FEATURE_COLUMNS = ["text_length_words", "engagement_total", "hashtag_count"]
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

POSITIVE_SENTIMENTS = {
    "acceptance",
    "accomplishment",
    "admiration",
    "adoration",
    "adrenaline",
    "adventure",
    "affection",
    "amazement",
    "amusement",
    "anticipation",
    "appreciation",
    "arousal",
    "awe",
    "artisticburst",
    "blessed",
    "breakthrough",
    "calmness",
    "captivation",
    "celebration",
    "celestial wonder",
    "charm",
    "colorful",
    "compassion",
    "compassionate",
    "confidence",
    "confident",
    "connection",
    "contentment",
    "creativity",
    "creative inspiration",
    "culinary adventure",
    "culinaryodyssey",
    "dazzle",
    "determination",
    "dreamchaser",
    "ecstasy",
    "elegance",
    "elation",
    "emotion",
    "empathetic",
    "empowerment",
    "enchantment",
    "energy",
    "engagement",
    "enjoyment",
    "enthusiasm",
    "envisioning history",
    "euphoria",
    "excitement",
    "exploration",
    "festivejoy",
    "free-spirited",
    "freedom",
    "friendship",
    "fulfillment",
    "grandeur",
    "grateful",
    "gratitude",
    "happy",
    "happiness",
    "harmony",
    "heartwarming",
    "hope",
    "hopeful",
    "hypnotic",
    "iconic",
    "imagination",
    "immersion",
    "inspiration",
    "inspired",
    "innerjourney",
    "journey",
    "joy",
    "joy in baking",
    "joyfulreunion",
    "kind",
    "kindness",
    "love",
    "marvel",
    "melodic",
    "mesmerizing",
    "mindfulness",
    "mischievous",
    "motivation",
    "nature's beauty",
    "nostalgia",
    "ocean's freedom",
    "optimism",
    "overjoyed",
    "playful",
    "playfuljoy",
    "positive",
    "positivity",
    "pride",
    "proud",
    "radiance",
    "reflection",
    "relief",
    "resilience",
    "rejuvenation",
    "reverence",
    "romance",
    "runway creativity",
    "satisfaction",
    "serenity",
    "solace",
    "spark",
    "success",
    "surprise",
    "sympathy",
    "tenderness",
    "thrill",
    "thrilling journey",
    "touched",
    "tranquility",
    "triumph",
    "vibrancy",
    "whimsy",
    "whispers of the past",
    "winter magic",
    "wonder",
    "wonderment",
    "zest",
}

NEGATIVE_SENTIMENTS = {
    "anger",
    "anxiety",
    "apprehensive",
    "bad",
    "betrayal",
    "bitter",
    "bitterness",
    "bittersweet",
    "boredom",
    "challenge",
    "darkness",
    "desolation",
    "despair",
    "desperation",
    "devastated",
    "disappointment",
    "disappointed",
    "disgust",
    "dismissive",
    "embarrassed",
    "emotionalstorm",
    "envious",
    "envy",
    "exhaustion",
    "fear",
    "fearful",
    "frustrated",
    "frustration",
    "grief",
    "hate",
    "heartache",
    "heartbreak",
    "helplessness",
    "intimidation",
    "isolation",
    "jealous",
    "jealousy",
    "loneliness",
    "loss",
    "lostlove",
    "melancholy",
    "miscalculation",
    "negative",
    "numbness",
    "obstacle",
    "overwhelmed",
    "pensive",
    "pressure",
    "regret",
    "resentment",
    "ruins",
    "sad",
    "sadness",
    "shame",
    "sorrow",
    "suffering",
    "suspense",
    "yearning",
}

NEUTRAL_SENTIMENTS = {
    "ambivalence",
    "confusion",
    "contemplation",
    "coziness",
    "curiosity",
    "indifference",
    "intrigue",
    "neutral",
    "renewed effort",
    "solitude",
}


def detect_delimiter(path: str | Path) -> str:
    header = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    semicolon_count = header.count(";")
    comma_count = header.count(",")
    return ";" if semicolon_count > comma_count else ","


def load_sentiment_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    delimiter = detect_delimiter(dataset_path)
    df = pd.read_csv(dataset_path, sep=delimiter)
    df.columns = [str(col).strip().lstrip(";") for col in df.columns]
    return df


def normalize_leetspeak(text: str) -> str:
    if not isinstance(text, str):
        return ""
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


def map_sentiment_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in POSITIVE_SENTIMENTS:
        return "positive"
    if normalized in NEGATIVE_SENTIMENTS:
        return "negative"
    if normalized in NEUTRAL_SENTIMENTS:
        return "neutral"
    return "neutral"


def prepare_training_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(col).strip().lstrip(";") for col in df.columns]
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"Dataset must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns. Found: {list(df.columns)}"
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df["cleaned_text"] = df[TEXT_COLUMN].fillna("").apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0].copy()
    df = df[df[LABEL_COLUMN].ne("")].copy()
    df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)

    df[TARGET_COLUMN] = df[LABEL_COLUMN].apply(map_sentiment_label)
    df["text_length_chars"] = df["cleaned_text"].str.len()
    df["text_length_words"] = df["cleaned_text"].str.split().str.len()

    for col in ["Retweets", "Likes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    df["engagement_total"] = df["Retweets"] + df["Likes"]
    if "Hashtags" in df.columns:
        df["Hashtags"] = df["Hashtags"].fillna("")
        df["hashtag_count"] = df["Hashtags"].astype(str).str.count("#")
    else:
        df["Hashtags"] = ""
        df["hashtag_count"] = 0
    return df


def build_inference_frame(cleaned_text: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cleaned_text": [cleaned_text],
            "text_length_words": [len(cleaned_text.split())],
            "engagement_total": [0],
            "hashtag_count": [0],
        }
    )


def force_single_thread(model):
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model
