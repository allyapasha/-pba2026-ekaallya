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

## Ringkasan Teknis

App ini memakai artefak model machine learning klasik, bukan deep learning:

- `sentiment_model.pkl`: `RandomForestClassifier`
- `tfidf_vectorizer.pkl`: `TfidfVectorizer`
- `label_encoder.pkl`: encoder label 3 kelas
- `scaler.pkl`: `StandardScaler` untuk fitur numerik

Alur inferensi:

- teks dibersihkan
- teks diubah ke fitur TF-IDF
- fitur numerik inferensi dibentuk dengan skema yang sama seperti training
- model mengembalikan probabilitas `positive`, `negative`, dan `neutral`

Catatan dataset:

- dataset sumber memakai delimiter `;`
- label emosi mentah pada dataset perlu diagregasi ke 3 kelas agar sesuai dengan output app

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
