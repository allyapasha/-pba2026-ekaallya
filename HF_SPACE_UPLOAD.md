# Upload Hugging Face Space

Deploy resmi sekarang memakai folder `apps/hf_space/`.

## Isi folder yang siap di-push

Folder `apps/hf_space/` sudah berisi:

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

Baseline neural di `artifacts/deep_learning/` tidak dipakai oleh Space.

## Langkah push manual

1. Login ke Hugging Face dan siapkan token akses.
2. Set token di terminal:

```powershell
$env:HF_TOKEN="hf_xxx"
```

3. Upload folder Space:

```powershell
python upload_to_hf_space.py `
  --username ekaallya `
  --space-name sentiment-analysis-indonesian `
  --folder .\apps\hf_space `
  --wait
```

4. Buka hasil deploy:

- Repo Space: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`
- App: `https://ekaallya-sentiment-analysis-indonesian.hf.space`

## Catatan

- Jangan hardcode token ke file Python. Gunakan `HF_TOKEN` atau `--token`.
- Jika artefak model di-refresh, salin ulang artefak terbaru ke `apps/hf_space/` sebelum upload.
- Folder deploy sekarang tidak lagi memakai snapshot lama `hf-space-sentiment-analysis-indonesian/`.
