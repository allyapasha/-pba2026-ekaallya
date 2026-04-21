from __future__ import annotations

from src.sentiment_project.inference import load_classic_artifacts
from apps.local.app import predict_sentiment
from validation.space_asset_check import main as check_space_assets


def main() -> None:
    print("Checking classic production artifacts...")
    load_classic_artifacts()

    samples = [
        "Saya sangat puas dengan produk ini",
        "Pelayanannya buruk dan mengecewakan",
        "Produk ini biasa saja",
    ]
    for text in samples:
        prediction = predict_sentiment(text)
        print(f"  {text} -> {prediction}")

    check_space_assets()
    print("Local system validation completed.")


if __name__ == "__main__":
    main()
