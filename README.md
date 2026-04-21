# Sentiment Analysis Indonesian

Repository ini berisi sistem analisis sentimen 3 kelas untuk teks bahasa Indonesia dengan jalur production yang sudah dipisahkan dari jalur eksperimen.

Output model selalu memakai tiga label:

- `positive`
- `negative`
- `neutral`

## Production Summary

Jalur production aktif memakai:

- model `LogisticRegression`
- fitur teks `TF-IDF` unigram dan bigram
- fitur numerik sederhana
- artefak yang dipaketkan untuk app lokal dan Hugging Face Space

Jalur ini dipilih sebagai default karena lebih ringan, lebih mudah dirawat, dan sudah terhubung ke app serta paket deploy.

## Repository Structure

```text
.
|-- src/sentiment_project/
|   `-- shared logic untuk preprocessing, training helper, config, dan inference
|-- pipelines/classic_ml/
|   `-- pipeline training production
|-- pipelines/deep_learning/
|   `-- baseline eksperimen non-production
|-- artifacts/classic_ml/
|   `-- artefak production aktif dan laporan evaluasi
|-- apps/local/
|   `-- app lokal production
|-- apps/hf_space/
|   `-- paket deploy production untuk Hugging Face Space
|-- docs/
|   `-- dokumentasi operasional dan audit yang relevan
|-- validation/
|   `-- smoke test lokal
|-- project-ml/
|   `-- folder legacy untuk dataset, notebook, dan wrapper kompatibilitas
|-- run_simple_pipeline.py
|   `-- entrypoint kompatibel untuk training production
`-- validate_local_system.py
    `-- entrypoint kompatibel untuk validasi lokal
```

## Active Production Flow

1. Training production dijalankan dari `python .\run_simple_pipeline.py`.
2. Artefak aktif disimpan di `artifacts/classic_ml/`.
3. App lokal memakai `apps/local/app.py`.
4. Paket deploy Hugging Face Space memakai `apps/hf_space/`.
5. Validasi ringan dijalankan dengan `python .\validate_local_system.py` atau `python -m validation.smoke_test`.

## Main Directories

### `src/sentiment_project/`
Pusat logika bersama untuk:

- konfigurasi path
- preprocessing teks
- mapping label
- training helper
- inference helper

### `pipelines/classic_ml/`
Pipeline training yang dipakai production. Wrapper root `run_simple_pipeline.py` mengarah ke sini.

### `pipelines/deep_learning/`
Baseline eksperimen untuk pembanding performa. Folder ini tidak menjadi jalur deploy default.

### `artifacts/classic_ml/`
Source of truth untuk artefak production:

- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `scaler.pkl`
- laporan evaluasi di `reports/`

### `apps/local/`
App lokal production untuk inferensi manual dan smoke testing cepat.

### `apps/hf_space/`
Folder deploy production untuk Hugging Face Space. Isi folder ini sudah disusun agar bisa dipush sebagai Space terpisah.

### `project-ml/`
Folder legacy yang dipertahankan untuk kebutuhan:

- dataset mentah dan hasil preprocessing
- notebook EDA dan modeling
- wrapper app lama

Folder ini bukan source of truth production untuk artefak atau deployment.

## Dataset

Dataset utama berada di:

- `project-ml/data/raw/sentimentdataset.csv`
- `sentimentdataset.csv` sebagai salinan root

Catatan penting:

- target sistem tetap 3 kelas
- dataset asli berisi label emosi granular yang dipetakan ke `positive`, `negative`, dan `neutral`
- audit singkat dataset tersedia di `docs/dataset_audit.md`

## Metrics

Model production aktif:

- accuracy: `0.8099`
- f1 weighted: `0.7951`
- f1 macro: `0.6857`

Baseline eksperimen:

- accuracy: `0.8451`
- f1 weighted: `0.8304`
- f1 macro: `0.7263`

Walau baseline eksperimen lebih tinggi, production tetap memakai model klasik karena jalur deploy dan operasionalnya lebih sederhana dan sudah stabil.

## Local Usage

Install dependency app:

```powershell
python -m pip install -r .\project-ml\app\requirements.txt
```

Refresh artefak production:

```powershell
python .\run_simple_pipeline.py
```

Jalankan smoke test:

```powershell
python .\validate_local_system.py
python -m validation.smoke_test
```

Jalankan app lokal:

```powershell
python .\apps\local\app.py
```

## Deployment

Deploy production untuk Hugging Face Space memakai folder:

- `apps/hf_space/`

Panduan deploy ringkas tersedia di `docs/deployment_hf_space.md`.

## Documentation

Dokumentasi yang dipertahankan untuk repo production:

- `docs/README.md`
- `docs/dataset_audit.md`
- `docs/classic_vs_deep_learning.md`
- `docs/deployment_hf_space.md`

## Status

Repository ini sudah disusun sebagai repo production-first.

- production path jelas
- eksperimen dipisahkan
- folder legacy diberi batas peran yang jelas
- dokumentasi internal proses kerja yang tidak relevan sudah dihapus dari root repo
