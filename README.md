# Sentiment Analysis Indonesian

Repositori ini telah dirapikan agar fokus pada proyek sentiment analysis teks berbahasa Indonesia.

## Struktur yang Dipertahankan

- `project-ml/`
  - `data/raw/sentimentdataset.csv`: dataset mentah
  - `data/processed/clean_data.csv`: hasil preprocessing
  - `models/`: artefak model sentiment analysis
  - `app/`: aplikasi Gradio untuk inferensi
  - `notebooks/`: notebook EDA dan modeling
- `sentimentdataset.csv`: salinan dataset di root
- `run_pipeline.py`: pipeline otomatis berbasis PyCaret
- `run_simple_pipeline.py`: pipeline sederhana berbasis scikit-learn
- `upload_to_hf_space.py`: script upload ke Hugging Face Space
- `HF_SPACE_UPLOAD.md`: panduan deploy ke Hugging Face

## Fokus Proyek

- Klasifikasi sentimen: `positive`, `negative`, `neutral`
- Preprocessing teks Indonesia
- Training dan evaluasi model
- Deploy model ke Hugging Face Space

## Menjalankan App Lokal

```powershell
pip install -r .\project-ml\app\requirements.txt
python .\project-ml\app\app.py
```

## Upload ke Hugging Face Space

Ikuti panduan di `HF_SPACE_UPLOAD.md`.

## Link Deploy

- Hugging Face Space: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`
- Aplikasi: `https://ekaallya-sentiment-analysis-indonesian.hf.space`
