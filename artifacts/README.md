# Artifacts

Folder `artifacts/` menyimpan output model dan laporan evaluasi.

## Subfolders

- `classic_ml/`
  Artefak production aktif dan source of truth untuk app production.
- `deep_learning/`
  Artefak eksperimen neural baseline. Folder ini dipertahankan terpisah agar eksperimen tidak mengganggu jalur production.

App production lokal dan Hugging Face Space mengambil artefak dari jalur production klasik.

## Catatan

- Artefak di `classic_ml/` boleh dianggap source of truth untuk deploy utama.
- Artefak di `deep_learning/` tidak otomatis dipakai production.
- Gunakan `python scripts/sync_space_assets.py` untuk menyalin artefak yang sudah valid ke folder `apps/hf_space/` dan `apps/hf_space_deep_learning/`.
