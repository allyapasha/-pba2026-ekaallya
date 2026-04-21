from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from .config import CLASSIC_ARTIFACTS_DIR, LEGACY_MODELS_DIR, REPO_ROOT
from .shared import (
    ARTIFACT_FILENAMES,
    NUMERIC_FEATURE_COLUMNS,
    build_keyword_prior,
    build_inference_frame,
    clean_text,
    force_single_thread,
)


def configure_repo_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def get_artifact_dirs(extra_dirs: list[Path] | None = None) -> list[Path]:
    candidates = [
        CLASSIC_ARTIFACTS_DIR,
        LEGACY_MODELS_DIR,
        Path.cwd() / "models",
        Path.cwd(),
    ]
    if extra_dirs:
        candidates = list(extra_dirs) + candidates

    ordered = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def resolve_artifact_path(filename: str, extra_dirs: list[Path] | None = None) -> Path:
    for directory in get_artifact_dirs(extra_dirs=extra_dirs):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in get_artifact_dirs(extra_dirs=extra_dirs))
    raise FileNotFoundError(f"{filename} not found. Searched: {searched}")


def load_classic_artifacts(extra_dirs: list[Path] | None = None):
    model = force_single_thread(joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["model"], extra_dirs=extra_dirs)))
    vectorizer = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["vectorizer"], extra_dirs=extra_dirs))
    label_encoder = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["label_encoder"], extra_dirs=extra_dirs))
    scaler = joblib.load(resolve_artifact_path(ARTIFACT_FILENAMES["scaler"], extra_dirs=extra_dirs))
    return model, vectorizer, label_encoder, scaler


def predict_with_classic_pipeline(text: str, artifacts=None) -> dict:
    if not text or not text.strip():
        return {"error": 1.0}

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"empty_after_cleaning": 1.0}

    if artifacts is None:
        artifacts = load_classic_artifacts()
    model, vectorizer, label_encoder, scaler = artifacts

    features = build_inference_frame(cleaned_text)
    text_features = vectorizer.transform(features["cleaned_text"])
    scaled_numeric = scaler.transform(features[NUMERIC_FEATURE_COLUMNS].to_numpy())
    model_input = np.hstack([text_features.toarray(), scaled_numeric])
    probabilities = model.predict_proba(model_input)[0]
    labels = label_encoder.inverse_transform(np.arange(len(probabilities)))
    result = {label: float(score) for label, score in zip(labels, probabilities)}

    keyword_prior = build_keyword_prior(cleaned_text)
    if keyword_prior:
        blend_weight = 0.45
        result = {
            label: (1 - blend_weight) * result[label] + blend_weight * keyword_prior.get(label, 0.0)
            for label in result
        }

    total = sum(result.values())
    if total > 0:
        result = {label: score / total for label, score in result.items()}
    return result
