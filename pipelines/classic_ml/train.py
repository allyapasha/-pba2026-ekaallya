from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment_project.classic_ml import save_classic_outputs, train_classic_model


def main() -> None:
    print("Running classic ML pipeline...")
    result = train_classic_model()
    save_classic_outputs(result)
    metrics = result["metrics"]
    print(json.dumps(
        {
            "model": metrics["model"],
            "accuracy": round(metrics["accuracy"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "class_distribution": metrics["class_distribution"],
        },
        indent=2,
    ))
    print("Classic artifacts refreshed.")


if __name__ == "__main__":
    main()

