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

Space ini memakai model produksi klasik dengan output tiga kelas: `positive`, `negative`, dan `neutral`.

## Model aktif

- Production: Logistic Regression + TF-IDF + fitur numerik sederhana
- Eksperimen: baseline neural ringan ada di `artifacts/deep_learning/`, tidak dipakai Space

## Isi minimum saat push ke Space

- `app.py`
- `README.md`
- `requirements.txt`
- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `scaler.pkl`

Artefak dapat diletakkan langsung di root Space atau di folder `models/`.

