# Pipelines

Folder `pipelines/` memisahkan training path berdasarkan tujuan penggunaan.

## Subfolders

- `classic_ml/`
  Pipeline training production aktif.
- `deep_learning/`
  Pipeline eksperimen pembanding, tidak dipakai default.

Wrapper root `run_simple_pipeline.py` dan `run_pipeline.py` sama-sama diarahkan ke pipeline production agar perintah lama tetap kompatibel.
