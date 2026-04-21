---
title: Sentiment Analysis Indonesian Deep Learning
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# Sentiment Analysis Indonesian Neural Experiment

Space ini adalah versi eksperimen neural baseline untuk sentiment analysis bahasa Indonesia.

## Space Link

- Space deep learning experiment: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian-deep-learning`
- Space machine learning production: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`

## Model Aktif

- `MLPClassifier`
- input teks dibersihkan dengan preprocessing Indonesia yang sama
- output tetap `positive`, `negative`, `neutral`
- file model di paket Space disimpan sebagai `sentiment_model.pkl` agar kompatibel dengan script upload

## Catatan

- Ini Space terpisah dari versi production klasik.
- Tujuannya untuk membandingkan baseline neural dengan model production utama.
- Environment lokal saat ini belum menyediakan framework seperti PyTorch atau TensorFlow.
- Karena itu jalur ini masih berupa baseline neural ringan, belum model sequence-aware seperti LSTM.
- Jalur production default tetap memakai versi machine learning klasik.
