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

# Sentiment Analysis Indonesian Deep Learning

Space ini adalah versi eksperimen deep learning ringan untuk sentiment analysis bahasa Indonesia.

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
- Jalur production default tetap memakai versi machine learning klasik.
