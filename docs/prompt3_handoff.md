# Handoff Setelah Prompt 2

## Status Repo

Repo sudah memiliki struktur yang lebih jelas untuk production dan eksperimen.

### Jalur production aktif

- training: `python run_simple_pipeline.py`
- validation: `python validate_local_system.py`
- app lokal: `python project-ml/app/app.py`
- deploy folder: `apps/hf_space/`

### Jalur eksperimen neural

- training: `python pipelines/deep_learning/train.py`
- output eksperimen: `artifacts/deep_learning/`

## Hal Yang Sudah Stabil

- output app tetap 3 kelas
- artefak production tersimpan di `artifacts/classic_ml/`
- folder Hugging Face Space sudah siap upload
- dokumentasi utama sudah diperbarui

## Risiko yang Masih Tersisa

- mapping label masih mengompresi banyak emosi granular ke 3 kelas
- kelas `neutral` masih kekurangan data
- baseline neural masih berupa MLP ringan, belum LSTM atau transformer
- penyesuaian keyword saat inferensi masih heuristik dan layak dievaluasi ulang bila dataset berubah

## Rekomendasi Prompt Berikutnya

1. audit manual label ambigu dan revisi mapping yang paling bermasalah
2. tambah dataset atau augmentasi untuk kelas `neutral`
3. coba baseline transformer kecil yang benar-benar sequence-aware
4. tambah test otomatis untuk path training, inferensi, dan deploy Space
5. bila baseline neural sudah matang, pertimbangkan jalur deploy terpisah untuk model neural
