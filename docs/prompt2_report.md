# Hasil Prompt 2

## Ringkasan

Prompt 2 menyelesaikan empat area utama:

1. audit dataset dan mapping label
2. peningkatan baseline production klasik
3. pemisahan struktur production vs experimentation
4. perapihan paket deploy Hugging Face Space

## Perubahan Utama

- Logika bersama dipindah ke `src/sentiment_project/`.
- Pipeline production dipisah ke `pipelines/classic_ml/`.
- Baseline neural dipisah ke `pipelines/deep_learning/`.
- Artefak production dipindah ke `artifacts/classic_ml/`.
- Artefak eksperimen neural dipindah ke `artifacts/deep_learning/`.
- App lokal production dipusatkan di `apps/local/app.py`.
- Paket deploy Hugging Face Space dipusatkan di `apps/hf_space/`.
- Entrypoint lama tetap hidup melalui wrapper:
  - `run_simple_pipeline.py`
  - `validate_local_system.py`
  - `project-ml/app/app.py`
  - `sentiment_system.py`

## Hasil Model Production

- Model lama Random Forest diganti ke Logistic Regression berimbang.
- Metrik production:
  - accuracy `0.8099`
  - f1 weighted `0.7951`
  - f1 macro `0.6857`
- App tetap menghasilkan probabilitas untuk `positive`, `negative`, `neutral`.

## Hasil Eksperimen Neural

- Baseline neural memakai `MLPClassifier`.
- Metrik:
  - accuracy `0.8451`
  - f1 weighted `0.8304`
  - f1 macro `0.7263`
- Status: eksperimen, belum dijadikan default production.

## Validasi Lokal

Smoke test berhasil untuk contoh berikut:

- `Saya sangat puas dengan produk ini`
- `Pelayanannya buruk dan mengecewakan`
- `Produk ini biasa saja`

## Deploy Hugging Face Space

- Folder deploy final: `apps/hf_space/`
- Artefak production terbaru sudah disalin ke folder tersebut
- Panduan deploy ada di `HF_SPACE_UPLOAD.md`

