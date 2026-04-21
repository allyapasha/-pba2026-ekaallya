---
title: Sentiment Analysis Indonesian
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# Sentiment Analysis Indonesian

Space ini memakai model production aktif dari repository utama dengan output tiga kelas:

- `positive`
- `negative`
- `neutral`

## Space Link

- Space machine learning production: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`

## Active Model

Model yang dipakai untuk deploy ini:

- `LogisticRegression`
- `TF-IDF`
- fitur numerik sederhana

Jalur deep learning tidak dipakai sebagai deploy default. Versi eksperimennya tersedia di:

- `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian-deep-learning`

## Minimum Files

Folder Space ini harus berisi:

- `app.py`
- `README.md`
- `requirements.txt`
- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `scaler.pkl`

Artefak boleh berada langsung di root Space atau di folder `models/`.
