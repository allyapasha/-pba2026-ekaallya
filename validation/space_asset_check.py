from __future__ import annotations

import hashlib
from pathlib import Path

from src.sentiment_project.config import CLASSIC_ARTIFACTS_DIR, DEEP_LEARNING_ARTIFACTS_DIR, HF_SPACE_DIR, REPO_ROOT
from src.sentiment_project.shared import ARTIFACT_FILENAMES


HF_SPACE_DEEP_DIR = REPO_ROOT / "apps" / "hf_space_deep_learning"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_file_pair(source: Path, target: Path, label: str) -> None:
    if not source.exists():
        raise FileNotFoundError(f"{label}: source missing {source}")
    if not target.exists():
        raise FileNotFoundError(f"{label}: target missing {target}")
    if sha256(source) != sha256(target):
        raise ValueError(f"{label}: file mismatch between {source} and {target}")


def check_classic_assets() -> None:
    for filename in ARTIFACT_FILENAMES.values():
        compare_file_pair(CLASSIC_ARTIFACTS_DIR / filename, HF_SPACE_DIR / filename, f"classic::{filename}")


def check_neural_assets() -> None:
    required = [
        ("mlp_model.pkl", "mlp_model.pkl"),
        ("mlp_model.pkl", "sentiment_model.pkl"),
        ("tfidf_vectorizer.pkl", "tfidf_vectorizer.pkl"),
        ("label_encoder.pkl", "label_encoder.pkl"),
        ("scaler.pkl", "scaler.pkl"),
    ]
    for source_name, target_name in required:
        compare_file_pair(
            DEEP_LEARNING_ARTIFACTS_DIR / source_name,
            HF_SPACE_DEEP_DIR / target_name,
            f"neural::{source_name}->{target_name}",
        )


def main() -> None:
    print("Checking production Space asset sync...")
    check_classic_assets()
    print("Production Space assets are synchronized.")

    if DEEP_LEARNING_ARTIFACTS_DIR.exists() and any(DEEP_LEARNING_ARTIFACTS_DIR.glob("*.pkl")):
        print("Checking neural experiment Space asset sync...")
        check_neural_assets()
        print("Neural experiment Space assets are synchronized.")
    else:
        print("Neural experiment artifacts not found. Skipping neural Space sync check.")


if __name__ == "__main__":
    main()
