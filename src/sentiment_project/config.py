from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "project-ml" / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "sentimentdataset.csv"
ROOT_DATASET_PATH = REPO_ROOT / "sentimentdataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_data.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CLASSIC_ARTIFACTS_DIR = ARTIFACTS_DIR / "classic_ml"
DEEP_LEARNING_ARTIFACTS_DIR = ARTIFACTS_DIR / "deep_learning"
CLASSIC_REPORTS_DIR = CLASSIC_ARTIFACTS_DIR / "reports"
DEEP_LEARNING_REPORTS_DIR = DEEP_LEARNING_ARTIFACTS_DIR / "reports"

LEGACY_MODELS_DIR = REPO_ROOT / "project-ml" / "models"
LOCAL_APP_DIR = REPO_ROOT / "apps" / "local"
HF_SPACE_DIR = REPO_ROOT / "apps" / "hf_space"
DOCS_DIR = REPO_ROOT / "docs"
VALIDATION_DIR = REPO_ROOT / "validation"


def ensure_repo_directories() -> None:
    for directory in [
        PROCESSED_DATA_PATH.parent,
        CLASSIC_ARTIFACTS_DIR,
        CLASSIC_REPORTS_DIR,
        DEEP_LEARNING_ARTIFACTS_DIR,
        DEEP_LEARNING_REPORTS_DIR,
        LEGACY_MODELS_DIR,
        DOCS_DIR,
        VALIDATION_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

