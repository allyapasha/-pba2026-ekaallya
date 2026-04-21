# Sentiment Analysis Indonesian

Repo ini berisi dua jalur yang dipisahkan jelas:

- jalur production `classic_ml` yang aktif untuk app lokal dan Hugging Face Space
- jalur eksperimen `deep_learning` untuk baseline neural yang tidak dipakai default

Output sistem tetap tiga kelas: `positive`, `negative`, dan `neutral`.

## Struktur Final

- `src/sentiment_project/`
  - modul bersama untuk preprocessing, mapping label, konfigurasi path, dan inferensi
- `pipelines/classic_ml/train.py`
  - training production path berbasis Logistic Regression
- `pipelines/deep_learning/train.py`
  - baseline neural ringan berbasis `MLPClassifier`
- `apps/local/app.py`
  - app lokal yang memakai model production
- `apps/hf_space/`
  - paket deploy Hugging Face Space yang siap di-push
- `artifacts/classic_ml/`
  - artefak model production dan laporan evaluasi
- `artifacts/deep_learning/`
  - output eksperimen neural bila training dijalankan manual
- `docs/`
  - audit dataset, laporan Prompt 2, perbandingan model, dan handoff lanjutan
- `validation/smoke_test.py`
  - validasi end-to-end ringan
- `project-ml/`
  - folder legacy yang dipertahankan untuk kompatibilitas dataset, notebook, dan app wrapper

## Model Production Aktif

Model production yang dipakai app lokal dan Hugging Face Space:

- `LogisticRegression` dengan `class_weight=balanced`
- `TfidfVectorizer` 1-2 gram
- fitur numerik sederhana
- penyesuaian probabilitas ringan berbasis keyword Indonesia umum saat inferensi

Alasan model ini dipilih:

- lebih seimbang dibanding Random Forest lama
- tetap kompatibel dengan kontrak output probabilitas 3 kelas
- artefak ringan dan mudah dipaketkan ke Hugging Face Space

## Hasil Evaluasi Ringkas

Model production:

- accuracy: `0.8099`
- f1 weighted: `0.7951`
- f1 macro: `0.6857`

Baseline neural:

- accuracy: `0.8451`
- f1 weighted: `0.8304`
- f1 macro: `0.7263`

Baseline neural belum dijadikan default karena jalur deploy dan audit inferensi production masih dipusatkan pada model klasik yang lebih sederhana dan lebih mudah dirawat.

## Jalankan Lokal

Install dependency:

```powershell
python -m pip install -r .\project-ml\app\requirements.txt
```

Refresh artefak production:

```powershell
python .\run_simple_pipeline.py
```

Jalankan baseline neural:

```powershell
python .\pipelines\deep_learning\train.py
```

Jalankan smoke test:

```powershell
python .\validate_local_system.py
```

Jalankan app lokal:

```powershell
python .\project-ml\app\app.py
```

## Hugging Face Space

Folder yang siap di-push:

- `apps/hf_space/`

Panduan langkah demi langkah ada di [HF_SPACE_UPLOAD.md](HF_SPACE_UPLOAD.md).

## Dokumen Penting

- `docs/dataset_audit.md`
- `docs/classic_vs_deep_learning.md`
- `docs/prompt2_report.md`
- `docs/prompt3_handoff.md`
