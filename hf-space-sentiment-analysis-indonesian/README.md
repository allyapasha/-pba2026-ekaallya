---
title: Sentiment Analysis Indonesian
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# Sentiment Analysis Indonesian

Gradio app untuk klasifikasi sentimen teks berbahasa Indonesia dengan tiga kelas: `positive`, `negative`, dan `neutral`.

## Isi folder Space

Pastikan folder yang di-upload ke Hugging Face Space berisi file berikut pada root:

- `app.py`
- `requirements.txt`
- `README.md`
- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `scaler.pkl`

`app.py` sudah dibuat fleksibel. Artefak model bisa diletakkan langsung di root Space atau di subfolder `models/`.
