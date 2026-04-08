# Upload Hugging Face Space

Script Python untuk membuat atau memperbarui Hugging Face Space untuk proyek sentiment analysis Indonesia.

## Siapkan folder Space

Buat folder terpisah yang berisi file deploy final, misalnya `hf-space-sentiment-analysis-indonesian/`, lalu salin file berikut ke root folder tersebut:

- `project-ml/app/app.py`
- `project-ml/app/requirements.txt`
- `project-ml/app/README.md`
- `project-ml/models/sentiment_model.pkl`
- `project-ml/models/tfidf_vectorizer.pkl`
- `project-ml/models/label_encoder.pkl`
- `project-ml/models/scaler.pkl`

Catatan penting:

- Nama file artefak di folder Space harus tetap sama seperti daftar di atas.
- `app.py` sekarang akan mencari artefak baik di root Space maupun di subfolder `models/`, jadi dua struktur itu aman.

## Jalankan upload

```powershell
$env:HF_TOKEN="hf_xxx"
python upload_to_hf_space.py `
  --username ekaallya `
  --space-name sentiment-analysis-indonesian `
  --folder .\hf-space-sentiment-analysis-indonesian `
  --wait
```

Contoh URL hasil:

- Repo Space: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`
- App: `https://ekaallya-sentiment-analysis-indonesian.hf.space`

Catatan:

- Jangan hardcode token ke file Python. Gunakan `HF_TOKEN` atau parameter `--token`.
- Script ini memakai `huggingface_hub`, jadi pastikan package tersebut terpasang.
