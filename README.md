# Sentiment Analysis Indonesian

Repository ini berisi sistem analisis sentimen 3 kelas untuk teks bahasa Indonesia dengan struktur production-first. Jalur machine learning dipakai sebagai deployment utama, sementara jalur deep learning dipertahankan sebagai baseline eksperimen yang terpisah.

## Penulis

| Nama | NIM |
| --- | --- |
| Allya Nurul Islami Pasha | 122450033 |
| Eka Fidiya Putri | 122450045 |

## Ringkasan Cepat

| Area | Jalur Utama |
| --- | --- |
| Output | `positive`, `negative`, `neutral` |
| Model production | `LogisticRegression` + `TF-IDF` + fitur numerik |
| App lokal | `apps/local/app.py` |
| Deploy production | `apps/hf_space/` |
| Training production | `pipelines/classic_ml/train.py` |
| Artefak production | `artifacts/classic_ml/` |
| Eksperimen | `pipelines/deep_learning/` |
| Folder legacy | `project-ml/` |

## Demo Online

| Jalur | Link |
| --- | --- |
| Machine learning production | `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian` |
| Deep learning experiment | `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian-deep-learning` |

Jalur machine learning adalah deploy utama. Jalur deep learning dipakai untuk pembanding eksperimen dan tidak menjadi default production.

## Arsitektur Model

| Jalur | Status | Model | Tujuan |
| --- | --- | --- | --- |
| Classic ML | Production | `LogisticRegression` | Dipakai untuk app lokal dan Hugging Face Space utama |
| Deep Learning | Experiment | `MLPClassifier` | Dipakai sebagai baseline eksperimen terpisah |

## Metrics

| Jalur | Accuracy | F1 Weighted | F1 Macro |
| --- | --- | --- | --- |
| Production classic ML | `0.8099` | `0.7951` | `0.6857` |
| Deep learning baseline | `0.8451` | `0.8304` | `0.7263` |

Walau baseline eksperimen memiliki metrik lebih tinggi, jalur production tetap memakai model klasik karena lebih ringan, lebih mudah dirawat, dan sudah stabil untuk deploy.

## Quick Start

Install dependency:

```powershell
python -m pip install -r .\project-ml\app\requirements.txt
```

Refresh artefak production:

```powershell
python .\run_simple_pipeline.py
```

Jalankan validasi ringan:

```powershell
python .\validate_local_system.py
python -m validation.smoke_test
```

Jalankan app lokal:

```powershell
python .\apps\local\app.py
```

## Struktur Repository

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
|-- scripts/
|   `-- utilitas operasional seperti upload ke Hugging Face Space
|-- validation/
|   `-- smoke test lokal
|-- project-ml/
|   `-- folder legacy untuk dataset, notebook, dan wrapper kompatibilitas
|-- run_simple_pipeline.py
|   `-- entrypoint kompatibel untuk training production
|-- run_pipeline.py
|   `-- alias lama yang kini diarahkan ke training production
|-- upload_to_hf_space.py
|   `-- wrapper kompatibilitas untuk script deploy
`-- validate_local_system.py
    `-- entrypoint kompatibel untuk validasi lokal
```

## Navigasi Folder

### `src/sentiment_project/`
Pusat logika bersama untuk:

- konfigurasi path
- preprocessing teks
- mapping label
- training helper
- inference helper

### `pipelines/classic_ml/`
Pipeline training yang dipakai production. Wrapper root `run_simple_pipeline.py` dan `run_pipeline.py` mengarah ke sini.

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

### `scripts/`
Tempat utilitas operasional yang tidak perlu memenuhi root repo. Wrapper root tetap dipertahankan hanya untuk kompatibilitas.

### `project-ml/`
Folder legacy yang dipertahankan untuk:

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

## Deployment

Deploy production untuk Hugging Face Space memakai folder:

- `apps/hf_space/`

Panduan deploy ringkas tersedia di `docs/deployment_hf_space.md`.

## Dokumentasi Terkait

- `docs/README.md`
- `docs/dataset_audit.md`
- `docs/classic_vs_deep_learning.md`
- `docs/deployment_hf_space.md`
- `apps/README.md`
- `pipelines/README.md`
- `artifacts/README.md`
- `scripts/README.md`

## Status Repository

Repository ini sudah disusun agar lebih rapi saat dibuka di GitHub.

- production path jelas
- eksperimen dipisahkan
- folder legacy diberi batas peran yang jelas
- script utilitas dipindahkan dari root ke folder yang lebih sesuai
- dokumentasi internal proses kerja yang tidak relevan sudah dihapus dari root repo


