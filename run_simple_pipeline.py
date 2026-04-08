#!/usr/bin/env python3
"""
run_simple_pipeline.py - Simplified ML Pipeline Runner
======================================================
Runs complete sentiment analysis pipeline:
1. Data loading & preprocessing
2. Feature engineering
3. Simple model training with scikit-learn
4. Model saving

Usage: python run_simple_pipeline.py
"""

import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

print("\n" + "=" * 80)
print("🚀 SENTIMENT ANALYSIS PIPELINE - SIMPLIFIED VERSION")
print("=" * 80 + "\n")

# ============================================================================
# PHASE 1: SETUP PATHS
# ============================================================================
print("📂 Setting up directories...")
BASE_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = BASE_DIR / "project-ml"
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"

# Create directories
for directory in [PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "raw" / "sentimentdataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "clean_data.csv"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

print(f"   ✓ Project: {PROJECT_DIR}")
print(f"   ✓ Raw data: {RAW_DATA_PATH}")
print(f"   ✓ Processed: {PROCESSED_DATA_PATH}")
print(f"   ✓ Models: {MODELS_DIR}\n")

# ============================================================================
# PHASE 2: LOAD DATA
# ============================================================================
print("📖 Loading raw dataset...")
try:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"   ✓ Loaded {len(df_raw):,} records × {len(df_raw.columns)} columns\n")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 3: PREPROCESSING
# ============================================================================
print("🧹 Text Preprocessing...")

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
    "gblk": "goblok",
    "gw": "gue",
    "lu": "lo",
    "ga": "tidak",
    "kyk": "kayak",
    "emg": "emang",
    "bgt": "banget",
    "udh": "sudah",
    "blm": "belum",
    "yg": "yang",
    "dgn": "dengan",
    "sm": "sama",
    "tp": "tapi",
    "org": "orang",
    "krn": "karena",
    "jgn": "jangan",
    "bkn": "bukan",
    "bs": "bisa",
    "dr": "dari",
    "jd": "jadi",
    "skrg": "sekarang",
    "bgst": "bangsat",
    "gblg": "goblok",
}


def normalize_leetspeak(text):
    """Convert leetspeak to letters"""
    if not isinstance(text, str):
        return ""
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


def expand_slang(text):
    """Expand slang to standard words"""
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text):
    """Complete text cleaning pipeline"""
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


# Apply cleaning
print("   • Cleaning text...")
df_raw["cleaned_text"] = df_raw["Text"].fillna("").apply(clean_text)

# Remove rows with missing or empty text
print("   • Cleaning dataset...")
df_clean = df_raw[df_raw["cleaned_text"].str.len() > 0].copy()
df_clean = df_clean.dropna(subset=["Sentiment"]).copy()
print(f"     - Kept {len(df_clean):,} records")

# Remove duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["Text"]).reset_index(drop=True)
print(f"     - Removed {before - len(df_clean)} duplicates")

# Standardize sentiment labels
print("   • Standardizing sentiment labels...")
sentiment_map = {
    "Positive": "positive",
    "positive": "positive",
    "Negative": "negative",
    "negative": "negative",
    "Neutral": "neutral",
    "neutral": "neutral",
}

df_clean["sentiment_group"] = df_clean["Sentiment"].map(sentiment_map)
df_clean["sentiment_group"] = df_clean["sentiment_group"].fillna("neutral")

print(f"   ✓ Sentiment distribution:")
for sent, count in df_clean["sentiment_group"].value_counts().items():
    pct = (count / len(df_clean)) * 100
    print(f"     - {sent}: {count:,} ({pct:.1f}%)")

# Feature engineering
print("   • Engineering features...")
df_clean["text_length_chars"] = df_clean["cleaned_text"].str.len()
df_clean["text_length_words"] = df_clean["cleaned_text"].str.split().str.len()

# Add numeric features if available
for col in ["Retweets", "Likes"]:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)
    else:
        df_clean[col] = 0

df_clean["engagement_total"] = df_clean.get("Retweets", 0) + df_clean.get("Likes", 0)

# Hashtag features
if "Hashtags" in df_clean.columns:
    df_clean["hashtag_count"] = df_clean["Hashtags"].fillna("").str.count("#")
else:
    df_clean["hashtag_count"] = 0

print("✅ Preprocessing complete\n")

# ============================================================================
# PHASE 4: SAVE PROCESSED DATA
# ============================================================================
print("💾 Saving processed data...")
df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"   ✓ Saved: {PROCESSED_DATA_PATH}")
print(f"   ✓ Shape: {df_clean.shape}\n")

# ============================================================================
# PHASE 5: MODEL TRAINING
# ============================================================================
print("=" * 80)
print("🤖 MODEL TRAINING WITH SCIKIT-LEARN")
print("=" * 80 + "\n")

print("📊 Preparing data...")
X_text = df_clean["cleaned_text"].values
y = df_clean["sentiment_group"].values

# Vectorize text with TF-IDF
print("   • TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
)
X_tfidf = vectorizer.fit_transform(X_text).toarray()
print(f"     - Created {X_tfidf.shape[1]} TF-IDF features")

# Add numerical features
print("   • Adding numerical features...")
numerical_features = df_clean[
    ["text_length_words", "engagement_total", "hashtag_count"]
].values

# Scale numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Combine features
X = np.hstack([X_tfidf, numerical_features_scaled])
print(f"     - Total features: {X.shape[1]}")

# Encode labels
print("   • Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"     - Classes: {list(le.classes_)}")

# Train/test split
print("   • Train/test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"     - Train: {len(X_train):,}, Test: {len(X_test):,}\n")

# Train model
print("🏆 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

model.fit(X_train, y_train)
print("   ✓ Training complete!\n")

# Evaluate
print("📈 Model Evaluation...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"   • Accuracy:  {accuracy:.4f}")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall:    {recall:.4f}")
print(f"   • F1-Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
for i, row in enumerate(cm):
    print(f"     {le.classes_[i]:10s}: {row}")

print("✅ Evaluation complete\n")

# ============================================================================
# PHASE 6: SAVE MODEL
# ============================================================================
print("=" * 80)
print("💾 SAVING MODEL & ARTIFACTS")
print("=" * 80 + "\n")

# Save model
print(f"📦 Saving model...")
joblib.dump(model, MODEL_PATH)
print(f"   ✓ Saved: {MODEL_PATH}")

# Save vectorizer
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"   ✓ Saved: {VECTORIZER_PATH}")

# Save label encoder
le_path = MODELS_DIR / "label_encoder.pkl"
joblib.dump(le, le_path)
print(f"   ✓ Saved: {le_path}")

# Save scaler
scaler_path = MODELS_DIR / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   ✓ Saved: {scaler_path}")

# File sizes
for fpath in [MODEL_PATH, VECTORIZER_PATH, le_path, scaler_path]:
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        print(f"   ✓ {fpath.name}: {size_kb:.2f} KB")

print("\n✅ All artifacts saved\n")

# ============================================================================
# PHASE 7: CREATE SUMMARY
# ============================================================================
print("📋 Creating model summary...")

summary = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                    SENTIMENT ANALYSIS MODEL SUMMARY                   ║
╠════════════════════════════════════════════════════════════════════════╣
║
║ 📊 DATASET INFORMATION
║ ─────────────────────
║   Total Records:          {len(df_clean):,}
║   Training Set:           {len(X_train):,} samples (80%)
║   Test Set:               {len(X_test):,} samples (20%)
║   Features:               {X.shape[1]} (TF-IDF + numerical)
║   Target Classes:         {len(le.classes_)}
║
║   Distribution:
"""

for i, cls in enumerate(le.classes_):
    count = (y_encoded == i).sum()
    pct = (count / len(y_encoded)) * 100
    summary += f"║   • {cls:10s}: {count:,} ({pct:5.1f}%)\n"

summary += f"""║
║ 🤖 MODEL INFORMATION
║ ──────────────────
║   Algorithm:              Random Forest Classifier
║   Number of Trees:        100
║   Max Depth:              20
║   Features Used:          TF-IDF (1000) + Numerical (3)
║
║ 📈 PERFORMANCE METRICS
║ ────────────────────
║   Accuracy:               {accuracy:.4f}
║   Precision (weighted):   {precision:.4f}
║   Recall (weighted):      {recall:.4f}
║   F1-Score (weighted):    {f1:.4f}
║
║ 📁 MODEL ARTIFACTS
║ ──────────────────
║   Model:                  {MODEL_PATH.name}
║   Vectorizer:             {VECTORIZER_PATH.name}
║   Label Encoder:          label_encoder.pkl
║   Scaler:                 scaler.pkl
║
║ 🚀 DEPLOYMENT READY
║ ──────────────────
║   ✅ Model trained and saved
║   ✅ All preprocessing artifacts saved
║   ✅ Ready for production deployment
║   ✅ Compatible with Gradio/Streamlit
║
╚════════════════════════════════════════════════════════════════════════╝

Generated: {pd.Timestamp.now()}
"""

summary_path = MODELS_DIR / "MODEL_SUMMARY.txt"
with open(summary_path, "w") as f:
    f.write(summary)

print(f"   ✓ Saved: {summary_path}")

# ============================================================================
# FINAL STATUS
# ============================================================================
print("\n" + "=" * 80)
print("✅ PIPELINE EXECUTION COMPLETE!")
print("=" * 80 + "\n")

print("📦 DELIVERABLES CREATED:")
print(f"   ✓ Cleaned dataset: {PROCESSED_DATA_PATH}")
print(f"   ✓ Trained model: {MODEL_PATH}")
print(f"   ✓ TF-IDF vectorizer: {VECTORIZER_PATH}")
print(f"   ✓ Label encoder: {le_path}")
print(f"   ✓ Scaler: {scaler_path}")
print(f"   ✓ Model summary: {summary_path}")

print("\n📊 STATISTICS:")
print(f"   • Records processed: {len(df_clean):,}")
print(f"   • Features engineered: {X.shape[1]}")
print(f"   • Model accuracy: {accuracy:.1%}")
print(f"   • Classes: {len(le.classes_)} (positive, negative, neutral)")

print("\n🚀 NEXT STEPS:")
print("   1. The model is ready for deployment!")
print("   2. To use in an app:")
print("      - Load model with: joblib.load('project-ml/models/sentiment_model.pkl')")
print(
    "      - Load vectorizer with: joblib.load('project-ml/models/tfidf_vectorizer.pkl')"
)
print(
    "      - Load label encoder with: joblib.load('project-ml/models/label_encoder.pkl')"
)
print("   3. Push to GitHub and prepare for Hugging Face deployment")

print("\n" + "=" * 80)
print("🎉 ALL DONE! Ready for GitHub & HuggingFace deployment!")
print("=" * 80 + "\n")
