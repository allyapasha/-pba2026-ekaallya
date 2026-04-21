#!/usr/bin/env python3
"""Backward-compatible wrapper for Hugging Face Space upload."""

from scripts.upload_to_hf_space import main


if __name__ == "__main__":
    raise SystemExit(main())
