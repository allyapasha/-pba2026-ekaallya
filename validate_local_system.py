#!/usr/bin/env python3
"""
Smoke test for the local Prompt 1 system state.
"""

import importlib.util
from pathlib import Path

import joblib

from sentiment_system import ARTIFACT_FILENAMES


BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "project-ml" / "models"
APP_PATH = BASE_DIR / "project-ml" / "app" / "app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("sentiment_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    print("Checking artifacts...")
    for filename in ARTIFACT_FILENAMES.values():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        obj = joblib.load(path)
        print(f"  OK {filename}: {type(obj).__name__}")

    print("Loading app module...")
    app = load_app_module()
    samples = [
        "Saya sangat puas dengan produk ini",
        "Pelayanannya buruk dan mengecewakan",
        "Produk ini biasa saja",
    ]
    for text in samples:
        prediction = app.predict_sentiment(text)
        print(f"  OK prediction for: {text}")
        print(f"     {prediction}")

    print("Local system validation completed.")


if __name__ == "__main__":
    main()
