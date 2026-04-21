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

## Ringkasan Model

Model yang aktif pada repo ini bukan deep learning. Pipeline utamanya memakai:

- `TfidfVectorizer` untuk representasi teks
- fitur numerik sederhana seperti panjang teks, total engagement, dan jumlah hashtag
- `StandardScaler` untuk penskalaan fitur numerik
- `RandomForestClassifier` untuk klasifikasi 3 kelas

Implementasi utamanya ada di `run_simple_pipeline.py`, sedangkan artefak inferensi yang dipakai app ada di `project-ml/models/`.

## Kesesuaian Dataset

Dataset `sentimentdataset.csv`:

- memakai delimiter `;`
- memiliki kolom utama `Text`, `Sentiment`, `Retweets`, `Likes`, dan `Hashtags`
- berisi banyak label emosi mentah yang perlu dipetakan ke 3 kelas output: `positive`, `negative`, `neutral`

Pipeline training sudah disesuaikan agar membaca delimiter `;` dan melakukan normalisasi label sebelum training.

## Menjalankan App Lokal

```powershell
pip install -r .\project-ml\app\requirements.txt
python .\project-ml\app\app.py
```

## Jalur Lokal Yang Direkomendasikan

1. Refresh artefak model:

```powershell
python .\run_simple_pipeline.py
```

2. Jalankan smoke test lokal:

```powershell
python .\validate_local_system.py
```

3. Jalankan app Gradio:

```powershell
python .\project-ml\app\app.py
```

## Upload ke Hugging Face Space

Ikuti panduan di `HF_SPACE_UPLOAD.md`.

## Link Deploy

- Hugging Face Space: `https://huggingface.co/spaces/ekaallya/sentiment-analysis-indonesian`
- Aplikasi: `https://ekaallya-sentiment-analysis-indonesian.hf.space`
