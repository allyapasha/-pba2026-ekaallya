# Sentiment Analysis Indonesian Text

End-to-end machine learning project untuk klasifikasi sentimen teks berbahasa Indonesia, mulai dari preprocessing data hingga deployment menggunakan Gradio.

---

## Overview

Proyek ini bertujuan untuk membangun sistem klasifikasi sentimen yang mampu mengelompokkan teks ke dalam tiga kategori: positive, negative, dan neutral.

Pipeline utama:

* Data preprocessing
* Feature engineering
* Modeling dengan PyCaret
* Evaluasi model
* Deployment dengan Gradio

---

## Project Structure

```
project-ml/
│
├── data/
│   ├── raw/
│   │   └── sentimentdataset.csv
│   └── processed/
│       └── clean_data.csv
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_modeling_pycaret.ipynb
│
├── models/
│   ├── sentiment_pycaret_best.pkl
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_auc.png
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── README.md
└── LICENSE
```

---

## Dataset

* Total data: 732
* Format: CSV
* Label: positive, negative, neutral

Kolom utama:

* text
* sentiment
* likes, retweets
* timestamp (year, month, day, hour)

Distribusi data:

* positive: 460
* negative: 190
* neutral: 82

---

## Workflow

### 1. Preprocessing

* Lowercase
* Remove URL dan mention
* Normalisasi teks
* Menghapus karakter khusus

### 2. Feature Engineering

* TF-IDF untuk teks
* Feature numerik (likes, retweets)

### 3. Modeling

* Menggunakan PyCaret
* Perbandingan beberapa model
* Hyperparameter tuning

### 4. Evaluation

* Accuracy
* Precision
* Recall
* F1-Score

### 5. Deployment

* Gradio interface
* Hugging Face Spaces

---

## Installation

### Requirements

* Python 3.10+

### Setup

```bash
git clone <repository-url>
cd project-ml

python -m venv venv
venv\Scripts\activate

pip install -r app/requirements.txt
```

---

## Usage

### Run Notebook

```bash
cd notebooks
jupyter notebook
```

### Run App

```bash
cd app
python app.py
```

Akses di browser:
[http://127.0.0.1:7860](http://127.0.0.1:7860)


## Model

Model dilatih menggunakan PyCaret dengan algoritma seperti:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Output model berupa file:

* sentiment_pycaret_best.pkl


## Deployment

Deploy menggunakan Hugging Face Spaces dengan Gradio.

Langkah:

1. Buat Space di Hugging Face
2. Upload file app.py dan requirements.txt
3. Tambahkan model
4. Jalankan aplikasi

## Author

* Allya Nurul Islami Pasha
* Eka Fidiya Putri
