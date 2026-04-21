from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment_project.deep_learning import save_neural_outputs, train_neural_baseline


def main() -> None:
    print("Running deep learning baseline...")
    result = train_neural_baseline()
    save_neural_outputs(result)
    metrics = result["metrics"]
    print(json.dumps(
        {
            "model": metrics["model"],
            "accuracy": round(metrics["accuracy"], 4),
            "f1_weighted": round(metrics["f1_weighted"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
        },
        indent=2,
    ))
    print("Deep learning artifacts refreshed.")


if __name__ == "__main__":
    main()
