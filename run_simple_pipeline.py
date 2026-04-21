#!/usr/bin/env python3
"""
run_simple_pipeline.py - Main local training pipeline

This is the canonical pipeline for Prompt 1 handoff:
1. Load the raw dataset
2. Normalize text and sentiment labels
3. Train a TF-IDF + Random Forest classifier
4. Save artifacts used by the local Gradio app
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sentiment_system import NUMERIC_FEATURE_COLUMNS, TARGET_COLUMN, load_sentiment_dataset, prepare_training_dataframe

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BASE_DIR / "project-ml"
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sentimentdataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_data.csv"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
SUMMARY_PATH = MODELS_DIR / "MODEL_SUMMARY.txt"


def ensure_directories() -> None:
    for directory in [PROCESSED_DATA_PATH.parent, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def build_summary(df_clean, X, X_train, X_test, classes, accuracy, precision, recall, f1) -> str:
    distribution = df_clean[TARGET_COLUMN].value_counts().to_dict()
    lines = [
        "SENTIMENT ANALYSIS MODEL SUMMARY",
        "",
        "DATASET:",
        f"  - Source file: {RAW_DATA_PATH}",
        f"  - Records: {len(df_clean)}",
        f"  - Features: {X.shape[1]}",
        f"  - Sentiment distribution: {distribution}",
        "",
        "PREPROCESSING:",
        "  - Shared module: sentiment_system.py",
        "  - Text cleaning: lowercase, URL removal, mention removal, leetspeak normalization, slang expansion, non-letter removal, whitespace normalization",
        "  - Label strategy: multi-emotion labels mapped into positive/negative/neutral",
        f"  - Numeric features: {NUMERIC_FEATURE_COLUMNS}",
        "",
        "MODEL:",
        "  - Algorithm: Random Forest",
        "  - Vectorizer: TF-IDF (1-2 grams, max_features=1000)",
        f"  - Classes: {classes}",
        f"  - Accuracy: {accuracy:.4f}",
        f"  - Precision: {precision:.4f}",
        f"  - Recall: {recall:.4f}",
        f"  - F1: {f1:.4f}",
        f"  - Train: {len(X_train)}, Test: {len(X_test)}",
        "",
        "FILES:",
        f"  - Cleaned data: {PROCESSED_DATA_PATH}",
        f"  - Model: {MODEL_PATH}",
        f"  - Vectorizer: {VECTORIZER_PATH}",
        f"  - Encoder: {ENCODER_PATH}",
        f"  - Scaler: {SCALER_PATH}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS PIPELINE - LOCAL CANONICAL VERSION")
    print("=" * 80 + "\n")

    ensure_directories()
    print(f"[1/6] Loading dataset from {RAW_DATA_PATH}")
    df_raw = load_sentiment_dataset(RAW_DATA_PATH)
    print(f"      Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns")

    print("[2/6] Preparing training dataframe")
    df_clean = prepare_training_dataframe(df_raw)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"      Saved cleaned data to {PROCESSED_DATA_PATH}")
    print(f"      Class distribution: {df_clean[TARGET_COLUMN].value_counts().to_dict()}")

    print("[3/6] Building features")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )
    X_text = df_clean["cleaned_text"].values
    X_tfidf = vectorizer.fit_transform(X_text).toarray()

    scaler = StandardScaler()
    numeric_features = df_clean[NUMERIC_FEATURE_COLUMNS].to_numpy()
    scaled_numeric = scaler.fit_transform(numeric_features)
    X = np.hstack([X_tfidf, scaled_numeric])

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_clean[TARGET_COLUMN].values)
    print(f"      TF-IDF features: {X_tfidf.shape[1]}")
    print(f"      Total features: {X.shape[1]}")
    print(f"      Classes: {list(encoder.classes_)}")

    print("[4/6] Training Random Forest model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    model.fit(X_train, y_train)

    print("[5/6] Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    print(f"      Accuracy:  {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall:    {recall:.4f}")
    print(f"      F1-score:  {f1:.4f}")
    print("      Confusion matrix:")
    for index, row in enumerate(cm):
        print(f"      {encoder.classes_[index]:10s}: {row}")

    print("[6/6] Saving artifacts")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    SUMMARY_PATH.write_text(
        build_summary(df_clean, X, X_train, X_test, list(encoder.classes_), accuracy, precision, recall, f1),
        encoding="utf-8",
    )
    print(f"      Saved model to {MODEL_PATH}")
    print(f"      Saved vectorizer to {VECTORIZER_PATH}")
    print(f"      Saved encoder to {ENCODER_PATH}")
    print(f"      Saved scaler to {SCALER_PATH}")
    print(f"      Saved summary to {SUMMARY_PATH}")

    print("\nPipeline complete. The local app can now use the refreshed artifacts.\n")


if __name__ == "__main__":
    main()
