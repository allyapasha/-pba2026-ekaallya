#!/usr/bin/env python3
"""Backward-compatible entrypoint for the production training pipeline."""

from pipelines.classic_ml.train import main


if __name__ == "__main__":
    main()
