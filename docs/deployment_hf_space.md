# Hugging Face Space Deployment

Deploy production resmi memakai folder `apps/hf_space/`.

## Folder yang Dideploy

Isi minimum folder Space:

- `app.py`
- `README.md`
- `requirements.txt`
- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `scaler.pkl`

Model yang dipakai Space adalah model production klasik:

- `LogisticRegression`
- `TF-IDF`
- fitur numerik sederhana
- penyesuaian probabilitas ringan berbasis keyword Indonesia umum saat inferensi

## Upload Manual

Sebelum upload, sinkronkan dulu paket Space dari source of truth artefak:

```powershell
python .\scripts\sync_space_assets.py
python -m validation.space_asset_check
```

Lalu upload folder Space yang sudah sinkron.

1. Login ke Hugging Face dan siapkan token akses.
2. Set token di terminal.
3. Upload folder `apps/hf_space/`.

```powershell
$env:HF_TOKEN="hf_xxx"

python .\upload_to_hf_space.py `
  --username ekaallya `
  --space-name sentiment-analysis-indonesian `
  --folder .\apps\hf_space `
  --wait
```

## Catatan

- jangan hardcode token di source code
- refresh artefak production sebelum upload bila model diperbarui
- jadikan `artifacts/classic_ml/` sebagai source of truth, bukan file yang ada langsung di folder Space
- folder `apps/hf_space_deep_learning/` bukan jalur deploy production default
