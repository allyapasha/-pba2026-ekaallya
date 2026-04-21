# App Lokal

Folder ini dipertahankan untuk kompatibilitas perintah lama:

```powershell
python .\project-ml\app\app.py
```

Implementasi utama app sekarang ada di `apps/local/app.py`.

## Model Aktif

- production: `LogisticRegression`
- fitur teks: `TF-IDF`
- fitur tambahan: panjang teks, engagement total, jumlah hashtag
- output: probabilitas `positive`, `negative`, `neutral`

## Catatan

- folder deploy Hugging Face Space sekarang ada di `apps/hf_space/`
- artefak production utama ada di `artifacts/classic_ml/`
- `project-ml/models/` bukan lagi source of truth production
