# Project ML

Folder ini berisi inti proyek sentiment analysis Indonesia.

## Isi Utama

- `data/raw/sentimentdataset.csv`: dataset mentah
- `data/processed/clean_data.csv`: hasil preprocessing
- `models/sentiment_model.pkl`: model klasifikasi final
- `models/tfidf_vectorizer.pkl`: vectorizer teks
- `models/label_encoder.pkl`: encoder label
- `models/scaler.pkl`: scaler fitur numerik
- `app/app.py`: aplikasi Gradio untuk inferensi
- `notebooks/01_eda_preprocessing.ipynb`: notebook preprocessing
- `notebooks/02_modeling_pycaret.ipynb`: notebook eksperimen modeling

## Menjalankan App

```powershell
pip install -r .\app\requirements.txt
python .\app\app.py
```

## Catatan

App memakai artefak yang ada di `models/`, jadi bisa dijalankan langsung dari struktur repo ini tanpa dependensi ke proyek lain.

Deploy aktif tersedia di Hugging Face Space:

- `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`
- `https://ekaallya-sentiment-analysis-indonesian.hf.space`
