from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DEEP_LEARNING_ARTIFACTS_DIR, DEEP_LEARNING_REPORTS_DIR, RAW_DATA_PATH, ensure_repo_directories
from .shared import NUMERIC_FEATURE_COLUMNS, TARGET_COLUMN, load_sentiment_dataset, prepare_training_dataframe


def train_neural_baseline():
    ensure_repo_directories()
    df_raw = load_sentiment_dataset(RAW_DATA_PATH)
    df_clean = prepare_training_dataframe(df_raw)

    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.9,
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_text = vectorizer.fit_transform(df_clean["cleaned_text"].values)

    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df_clean[NUMERIC_FEATURE_COLUMNS].to_numpy())
    X = np.hstack([X_text.toarray(), scaled_numeric])

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_clean[TARGET_COLUMN].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "model": "MLPClassifier",
        "classes": list(encoder.classes_),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=list(encoder.classes_),
            zero_division=0,
            output_dict=True,
        ),
    }
    return {
        "model": model,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "scaler": scaler,
        "metrics": metrics,
    }


def save_neural_outputs(result: dict) -> None:
    ensure_repo_directories()
    joblib.dump(result["model"], DEEP_LEARNING_ARTIFACTS_DIR / "mlp_model.pkl")
    joblib.dump(result["vectorizer"], DEEP_LEARNING_ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(result["encoder"], DEEP_LEARNING_ARTIFACTS_DIR / "label_encoder.pkl")
    joblib.dump(result["scaler"], DEEP_LEARNING_ARTIFACTS_DIR / "scaler.pkl")
    (DEEP_LEARNING_REPORTS_DIR / "deep_learning_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    (DEEP_LEARNING_REPORTS_DIR / "deep_learning_notes.md").write_text(
        "# Baseline Deep Learning\n\n"
        "Eksperimen Prompt 2 memakai `MLPClassifier` sebagai baseline neural ringan.\n\n"
        "- Jalur ini dipisahkan dari production dan tidak dipakai oleh app default.\n"
        "- Pendekatan ini belum sequence-aware seperti LSTM atau transformer.\n"
        "- Tujuannya memberi baseline neural yang murah untuk dijalankan lokal sebelum migrasi ke model yang lebih berat.\n",
        encoding="utf-8",
    )
