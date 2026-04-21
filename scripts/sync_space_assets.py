from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment_project.config import CLASSIC_ARTIFACTS_DIR, DEEP_LEARNING_ARTIFACTS_DIR, HF_SPACE_DIR
from src.sentiment_project.shared import ARTIFACT_FILENAMES


HF_SPACE_DEEP_DIR = REPO_ROOT / "apps" / "hf_space_deep_learning"
CLASSIC_REPORTS_DIR = CLASSIC_ARTIFACTS_DIR / "reports"
DEEP_REPORTS_DIR = DEEP_LEARNING_ARTIFACTS_DIR / "reports"


def copy_required_files(source_dir: Path, target_dir: Path, filenames: list[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        shutil.copy2(source, target_dir / filename)


def write_manifest(target_dir: Path, payload: dict) -> None:
    (target_dir / "asset_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sync_classic_assets() -> None:
    copy_required_files(CLASSIC_ARTIFACTS_DIR, HF_SPACE_DIR, list(ARTIFACT_FILENAMES.values()))
    metrics_path = CLASSIC_REPORTS_DIR / "classic_metrics.json"
    payload = {
        "package": "hf_space_production",
        "model_family": "classic_ml",
        "artifacts_source": str(CLASSIC_ARTIFACTS_DIR),
        "metrics_source": str(metrics_path),
    }
    write_manifest(HF_SPACE_DIR, payload)


def sync_neural_assets() -> None:
    deep_files = ["mlp_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl", "scaler.pkl"]
    copy_required_files(DEEP_LEARNING_ARTIFACTS_DIR, HF_SPACE_DEEP_DIR, deep_files)
    # Compatibility alias for the Space app loader that expects sentiment_model.pkl.
    shutil.copy2(DEEP_LEARNING_ARTIFACTS_DIR / "mlp_model.pkl", HF_SPACE_DEEP_DIR / "sentiment_model.pkl")
    metrics_path = DEEP_REPORTS_DIR / "deep_learning_metrics.json"
    payload = {
        "package": "hf_space_neural_experiment",
        "model_family": "neural_baseline",
        "artifacts_source": str(DEEP_LEARNING_ARTIFACTS_DIR),
        "metrics_source": str(metrics_path),
        "compatibility_alias": "sentiment_model.pkl -> mlp_model.pkl",
    }
    write_manifest(HF_SPACE_DEEP_DIR, payload)


def main() -> None:
    sync_classic_assets()
    print(f"Production Space assets synced to {HF_SPACE_DIR}")
    if DEEP_LEARNING_ARTIFACTS_DIR.exists() and any(DEEP_LEARNING_ARTIFACTS_DIR.glob('*.pkl')):
        sync_neural_assets()
        print(f"Neural experiment Space assets synced to {HF_SPACE_DEEP_DIR}")
    else:
        print("Neural experiment artifacts not found. Skipping deep learning Space sync.")


if __name__ == "__main__":
    main()
