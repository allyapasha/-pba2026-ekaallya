from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CLASSIC_ARTIFACTS_DIR, CLASSIC_REPORTS_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, ensure_repo_directories
from .shared import ARTIFACT_FILENAMES, NUMERIC_FEATURE_COLUMNS, TARGET_COLUMN, load_sentiment_dataset, prepare_training_dataframe


@dataclass
class ClassicArtifacts:
    model_path: Path
    vectorizer_path: Path
    encoder_path: Path
    scaler_path: Path
    summary_path: Path
    metrics_path: Path
    report_path: Path


def get_classic_artifact_paths() -> ClassicArtifacts:
    return ClassicArtifacts(
        model_path=CLASSIC_ARTIFACTS_DIR / ARTIFACT_FILENAMES["model"],
        vectorizer_path=CLASSIC_ARTIFACTS_DIR / ARTIFACT_FILENAMES["vectorizer"],
        encoder_path=CLASSIC_ARTIFACTS_DIR / ARTIFACT_FILENAMES["label_encoder"],
        scaler_path=CLASSIC_ARTIFACTS_DIR / ARTIFACT_FILENAMES["scaler"],
        summary_path=CLASSIC_ARTIFACTS_DIR / "MODEL_SUMMARY.txt",
        metrics_path=CLASSIC_REPORTS_DIR / "classic_metrics.json",
        report_path=CLASSIC_REPORTS_DIR / "classic_evaluation.md",
    )


def train_classic_model():
    ensure_repo_directories()
    df_raw = load_sentiment_dataset(RAW_DATA_PATH)
    df_clean = prepare_training_dataframe(df_raw)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

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
    numeric_features = df_clean[NUMERIC_FEATURE_COLUMNS].to_numpy()
    scaled_numeric = scaler.fit_transform(numeric_features)
    X = np.hstack([X_text.toarray(), scaled_numeric])

    encoder = LabelEncoder()
    y = encoder.fit_transform(df_clean[TARGET_COLUMN].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        C=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = list(encoder.classes_)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0, output_dict=True)

    metrics = {
        "model": "LogisticRegression",
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": 3000,
            "min_df": 2,
            "max_df": 0.9,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
        "class_distribution": df_clean[TARGET_COLUMN].value_counts().to_dict(),
        "classes": labels,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    return {
        "df_clean": df_clean,
        "model": model,
        "vectorizer": vectorizer,
        "encoder": encoder,
        "scaler": scaler,
        "metrics": metrics,
    }


def build_classic_summary(metrics: dict) -> str:
    lines = [
        "SENTIMENT ANALYSIS MODEL SUMMARY",
        "",
        "MODEL PRODUCTION:",
        "  - Algorithm: Logistic Regression",
        "  - Strategy: class_weight=balanced",
        "  - Vectorizer: TF-IDF 1-2 gram, max_features=3000, sublinear_tf=True",
        f"  - Numeric features: {NUMERIC_FEATURE_COLUMNS}",
        f"  - Classes: {metrics['classes']}",
        "",
        "DATASET:",
        f"  - Source file: {RAW_DATA_PATH}",
        f"  - Records after cleaning: {sum(metrics['class_distribution'].values())}",
        f"  - Sentiment distribution: {metrics['class_distribution']}",
        "",
        "METRICS:",
        f"  - Accuracy: {metrics['accuracy']:.4f}",
        f"  - Precision weighted: {metrics['precision_weighted']:.4f}",
        f"  - Recall weighted: {metrics['recall_weighted']:.4f}",
        f"  - F1 weighted: {metrics['f1_weighted']:.4f}",
        f"  - F1 macro: {metrics['f1_macro']:.4f}",
        f"  - Train size: {metrics['train_size']}",
        f"  - Test size: {metrics['test_size']}",
    ]
    return "\n".join(lines) + "\n"


def build_classic_evaluation_markdown(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    return (
        "# Evaluasi Model Klasik\n\n"
        "## Ringkasan\n"
        f"- Model: `{metrics['model']}`\n"
        f"- Accuracy: `{metrics['accuracy']:.4f}`\n"
        f"- F1 weighted: `{metrics['f1_weighted']:.4f}`\n"
        f"- F1 macro: `{metrics['f1_macro']:.4f}`\n"
        f"- Distribusi kelas: `{metrics['class_distribution']}`\n\n"
        "## Confusion Matrix\n"
        "| actual \\ predicted | negative | neutral | positive |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| negative | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |\n"
        f"| neutral | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |\n"
        f"| positive | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |\n\n"
        "## Analisis\n"
        "- Model sebelumnya bias ke `positive` karena distribusi label sangat timpang dan Random Forest cenderung mengikuti kelas mayoritas.\n"
        "- Logistic Regression dengan `class_weight=balanced` menaikkan recall kelas `negative` dan `neutral` tanpa merusak kontrak probabilitas 3 kelas.\n"
        "- Kelas `neutral` masih paling sulit karena jumlah contoh sedikit dan semantik beberapa label emosi ambigu.\n"
    )


def save_classic_outputs(result: dict) -> ClassicArtifacts:
    paths = get_classic_artifact_paths()
    joblib.dump(result["model"], paths.model_path)
    joblib.dump(result["vectorizer"], paths.vectorizer_path)
    joblib.dump(result["encoder"], paths.encoder_path)
    joblib.dump(result["scaler"], paths.scaler_path)
    paths.summary_path.write_text(build_classic_summary(result["metrics"]), encoding="utf-8")
    paths.metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    paths.report_path.write_text(build_classic_evaluation_markdown(result["metrics"]), encoding="utf-8")
    return paths
