# 📊 Sentiment Analysis - End-to-End ML Project

![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-yellow)
![Gradio](https://img.shields.io/badge/Gradio-Interface-ff69b4)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Daftar Isi
1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Anggota Tim](#anggota-tim)
3. [Struktur Repository](#struktur-repository)
4. [Dataset](#dataset)
5. [Alur Kerja (Workflow)](#alur-kerja-workflow)
6. [Setup & Instalasi](#setup--instalasi)
7. [Cara Penggunaan](#cara-penggunaan)
8. [Model & Performa](#model--performa)
9. [Deployment](#deployment)
10. [Referensi](#referensi)

---

## 🎯 Deskripsi Proyek

Proyek ini bertujuan untuk menyelesaikan **klasifikasi sentimen (Sentiment Analysis)** pada teks berbahasa Indonesia secara **end-to-end**, mulai dari tahap preprocessing data hingga deployment di Hugging Face Spaces.

### Tujuan Utama
- ✅ Melakukan EDA (Exploratory Data Analysis) pada dataset sentimen
- ✅ Membersihkan dan mempersiapkan data dengan standar `mct-nlp`
- ✅ Melatih model menggunakan PyCaret AutoML
- ✅ Melakukan hyperparameter tuning untuk optimasi performa
- ✅ Deploy model ke Hugging Face Spaces menggunakan Gradio
- ✅ Membuat dokumentasi lengkap dan insights

### Pendekatan
- **Data Preprocessing**: Mengikuti standar dari `mct-nlp` (lowercase, URL removal, leetspeak normalization, slang expansion)
- **Feature Engineering**: Text vectorization dengan TF-IDF, numerical feature scaling
- **Model Selection**: Automated model comparison dengan PyCaret
- **Deployment**: Gradio interface dengan template dari `deteksi-toksisitas-chat`

---

## 👥 Anggota Tim

| Nama | NIM | GitHub | Role |
|------|-----|--------|------|
| Allya Nurul Islami Pasha | 122450033 | @allyapasha | Person 1 - Data Analyst (Preprocessing) |
| Eka Fidiya Putri | 122450045 | @eka409 | Person 2 - ML Engineer (Modeling & Deployment) |

---

## 📁 Struktur Repository

```
project-ml/
├── 📂 data/
│   ├── raw/
│   │   └── sentimentdataset.csv          # Dataset mentah (732 samples)
│   └── processed/
│       └── clean_data.csv                # Data bersih setelah preprocessing
│
├── 📂 notebooks/
│   ├── 01_eda_preprocessing.ipynb        # EDA & preprocessing (Person 1)
│   └── 02_modeling_pycaret.ipynb         # PyCaret modeling (Person 2)
│
├── 📂 models/
│   ├── sentiment_pycaret_best.pkl        # Final trained model
│   ├── confusion_matrix.png              # Performance visualization
│   ├── feature_importance.png            # Feature importance plot
│   ├── roc_auc.png                       # ROC AUC curve
│   └── MODEL_SUMMARY.txt                 # Model documentation
│
├── 📂 app/
│   ├── app.py                            # Gradio deployment app
│   ├── requirements.txt                  # Python dependencies
│   └── README.md                         # App-specific documentation
│
├── README.md                             # This file
├── .gitignore
└── LICENSE

```

---

## 📊 Dataset

### Informasi Dataset
- **Sumber**: `sentimentdataset.csv`
- **Jumlah Record**: 732 samples (sebelum cleaning)
- **Format**: CSV dengan 14 kolom

### Kolom Dataset
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `text` | string | Teks mentah dari social media |
| `sentiment` | string | Label sentimen (Positive/Negative/Neutral) |
| `timestamp` | string | Waktu posting |
| `user` | string | Username pengguna |
| `platform` | string | Platform (Twitter/Instagram/Facebook) |
| `country` | string | Negara pengguna |
| `hashtags` | string | Hashtag dalam teks |
| `retweets` | float | Jumlah retweets |
| `likes` | float | Jumlah likes |
| `year`, `month`, `day`, `hour` | int | Temporal features |

### Distribusi Target
Setelah cleaning:
```
positive: 460 samples (62.8%)
negative: 190 samples (25.9%)
neutral:   82 samples (11.2%)
────────────────────────────
total:     732 samples
```

---

## 🔄 Alur Kerja (Workflow)

### Fase 1: Analisis & Planning
1. **Sync-up Meeting**: Person 1 & 2 mempelajari:
   - Standar koding dari `mct-nlp` (config.py, preprocess.py)
   - Template deployment dari `deteksi-toksisitas-chat` (app.py, requirements.txt)

2. **Observasi Dataset**:
   - Memahami struktur dan karakteristik `sentimentdataset.csv`
   - Mengidentifikasi missing values dan duplicates

### Fase 2: Data Preprocessing (Person 1)
**Notebook**: `01_eda_preprocessing.ipynb`

1. **EDA (Exploratory Data Analysis)**
   - Analisis distribusi data
   - Identifikasi missing values dan outliers
   - Visualisasi target variable distribution

2. **Data Cleaning**
   - Menghapus duplikasi
   - Menangani missing values
   - Standardisasi format kolom

3. **Text Preprocessing** (mct-nlp style)
   - Lowercase conversion
   - URL dan mention removal
   - Leetspeak normalization (0→o, 1→i, dst)
   - Slang expansion (gw→gue, lu→lo, dst)
   - Punctuation & special character removal
   - Whitespace normalization

4. **Feature Engineering**
   - Text length features (chars, words)
   - Engagement metrics (retweets, likes, total)
   - Temporal features (year, month, day, hour)
   - Scaling & normalization

5. **Output**
   - `clean_data.csv` (732 rows × 33 columns)
   - Visualization & insights

### Fase 3: Modeling (Person 2)
**Notebook**: `02_modeling_pycaret.ipynb`

1. **PyCaret Setup**
   - Load `clean_data.csv`
   - Configure setup (train/test split 80/20, 5-fold CV)
   - Automatic TF-IDF vectorization untuk text features

2. **Model Comparison**
   - Jalankan `compare_models()`
   - Benchmark multiple algorithms:
     - Logistic Regression (LR)
     - Random Forest (RF)
     - Gradient Boosting (GB)
     - XGBoost (XGB)
     - LightGBM (LGB)
     - dll.

3. **Model Selection**
   - Pilih top 3 models berdasarkan Accuracy
   - Select best model untuk tuning

4. **Hyperparameter Tuning**
   - Gunakan `tune_model()` dengan RandomizedSearchCV
   - Optimize berdasarkan metric: Accuracy

5. **Model Finalization**
   - Train final model on full dataset
   - Evaluate performance:
     - Accuracy, Precision, Recall, F1-Score
     - Confusion Matrix
     - ROC AUC Curve
   - Save as `.pkl`

6. **Output**
   - `sentiment_pycaret_best.pkl` (trained model)
   - Performance visualizations
   - Model summary documentation

### Fase 4: Deployment (Person 2)
**Folder**: `app/`

1. **Create Gradio App** (`app.py`)
   - Load model dari `.pkl`
   - Implement text preprocessing (sama dengan training)
   - Create prediction function
   - Build Gradio interface dengan examples
   - Styling & user experience

2. **Prepare Requirements** (`requirements.txt`)
   - PyCaret, Gradio, Pandas, scikit-learn, numpy
   - Version specifications untuk reproducibility

3. **Deploy to Hugging Face Spaces**
   - Create new Space: `sentiment-analysis-indonesian`
   - Upload files: `app.py`, `requirements.txt`, model pkl
   - Configure: Python 3.10, Gradio
   - Launch dan test

4. **Output**
   - Live demo link: [HuggingFace Spaces]
   - Web-based interface untuk predictions

### Fase 5: Dokumentasi & Finalisasi
1. **Update README.md**
   - Project overview
   - Setup instructions
   - Usage examples
   - Model performance insights
   - Deployment link

2. **Code Documentation**
   - Docstrings pada setiap function
   - Comments untuk logika kompleks
   - Type hints untuk clarity

3. **Presentations**
   - Summary findings
   - Model performance comparison
   - Deployment walkthrough

---

## 🚀 Setup & Instalasi

### Prerequisites
- Python 3.10+
- pip atau conda
- Git (untuk cloning repo)

### Local Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd project-ml
```

2. **Create Virtual Environment** (recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sentiment-ml python=3.10
conda activate sentiment-ml
```

3. **Install Dependencies**

Untuk development (full):
```bash
pip install -r requirements_dev.txt
```

Untuk production/deployment:
```bash
pip install -r app/requirements.txt
```

Minimal installation:
```bash
pip install pycaret pandas gradio scikit-learn numpy
```

4. **Verify Installation**
```bash
python -c "import pycaret; import gradio; print('✅ Installation successful!')"
```

---

## 📖 Cara Penggunaan

### 1. Running Notebooks

#### Notebook 1: EDA & Preprocessing
```bash
cd notebooks
jupyter notebook 01_eda_preprocessing.ipynb
```
**Hasil**:
- Visualisasi data distribution
- Cleaned dataset: `data/processed/clean_data.csv`
- Summary statistics

#### Notebook 2: Modeling & PyCaret
```bash
cd notebooks
jupyter notebook 02_modeling_pycaret.ipynb
```
**Hasil**:
- Trained model: `models/sentiment_pycaret_best.pkl`
- Performance metrics & plots
- Model summary documentation

### 2. Running Gradio App (Local)

```bash
cd app
python app.py
```

**Output**:
```
🚀 Starting Gradio App for Sentiment Analysis
============================================================
📱 Local URL: http://127.0.0.1:7860
🌐 Share link will be available after launch
============================================================
```

Buka browser ke `http://127.0.0.1:7860` untuk mengakses interface.

### 3. Example Predictions

**Positive Example**:
```
Input: "Saya sangat senang dengan produk ini! Kualitasnya luar biasa!"
Output: Positive (confidence: 0.92)
```

**Negative Example**:
```
Input: "Pelayanan mereka sangat buruk dan kecewa"
Output: Negative (confidence: 0.88)
```

**Neutral Example**:
```
Input: "Produk ini biasa saja, tidak ada yang istimewa"
Output: Neutral (confidence: 0.75)
```

---

## 🎯 Model & Performa

### Model Architecture

```
Input Text
    ↓
Text Preprocessing Pipeline
├── Lowercase
├── URL/Mention Removal
├── Leetspeak Normalization
├── Slang Expansion
└── Character Cleaning
    ↓
TF-IDF Vectorization (Auto via PyCaret)
    ↓
Feature Scaling & Engineering
├── Numerical features normalization
├── Text length features
├── Engagement metrics
└── Temporal features
    ↓
Classification Model (PyCaret Best)
    ├── Logistic Regression
    ├── Random Forest
    ├── Gradient Boosting
    ├── XGBoost
    └── LightGBM
    ↓
Sentiment Prediction
└── Output: {Positive: 0.85, Negative: 0.10, Neutral: 0.05}
```

### Performance Metrics

**Training Setup**:
- Train/Test Split: 80/20
- Cross-Validation: 5-fold stratified
- Optimization Metric: Accuracy

**Model Comparison Results** (Top 3):
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Model 1 | TBD | TBD | TBD | TBD |
| Model 2 | TBD | TBD | TBD | TBD |
| Model 3 | TBD | TBD | TBD | TBD |

*Note: Metrics akan diupdate setelah running notebook*

### Feature Importance
Top features yang paling berkontribusi pada prediksi:
1. `cleaned_text` (TF-IDF features)
2. `text_length_words`
3. `engagement_total`
4. `likes_scaled`
5. `cleaned_text_length_chars`

### Model Files
```
models/
├── sentiment_pycaret_best.pkl          # Main model (trained)
├── sentiment_pycaret_best.pkl.zip      # Compressed version
├── confusion_matrix.png                # Test set performance
├── feature_importance.png              # Top features
├── roc_auc.png                         # ROC curve (multiclass)
└── MODEL_SUMMARY.txt                   # Text documentation
```

---

## 🌐 Deployment

### Hugging Face Spaces

**Space**: [sentiment-analysis-indonesian](https://huggingface.co/spaces/username/sentiment-analysis-indonesian)

**Configuration**:
```yaml
title: Sentiment Analysis - Indonesian Text
emoji: 😊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.20.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
```

### Deployment Steps

1. **Create Space di Hugging Face**
   - Go to https://huggingface.co/new-space
   - Select: Gradio + Python 3.10

2. **Upload Files**
   ```bash
   cd project-ml/app
   git clone https://huggingface.co/spaces/username/sentiment-analysis-indonesian
   cp app.py requirements.txt sentiment-analysis-indonesian/
   cp ../models/sentiment_pycaret_best.pkl sentiment-analysis-indonesian/
   cd sentiment-analysis-indonesian
   git add .
   git commit -m "Initial commit: Sentiment Analysis app"
   git push
   ```

3. **Verify Deployment**
   - Check Space URL
   - Test predictions via web interface
   - Share link dengan stakeholders

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY app/app.py .
COPY models/sentiment_pycaret_best.pkl .

EXPOSE 7860
CMD ["python", "app.py"]
```

Build & run:
```bash
docker build -t sentiment-analysis .
docker run -p 7860:7860 sentiment-analysis
```

---

## 📚 Referensi & Inspirasi

### Reference Folders
1. **`mct-nlp/`**: Standar koding & preprocessing
   - `module_ML/config.py` - Configuration management
   - `module_ML/preprocess.py` - Text preprocessing pipeline
   - `module_ML/train.py` - Model training script

2. **`deteksi-toksisitas-chat/`**: Deployment template
   - `app.py` - Gradio interface structure
   - `requirements.txt` - Dependency specification
   - Hugging Face Spaces configuration

### Libraries & Tools
- **PyCaret**: Automated ML & model comparison
- **Gradio**: Web interface untuk predictions
- **Scikit-learn**: Machine learning algorithms
- **Pandas & NumPy**: Data manipulation
- **Hugging Face**: Model hosting & deployment

### Indonesian NLP Resources
- [IndonLU Dataset](https://huggingface.co/datasets/indonlp/indonlu)
- [Sastrawi - Indonesian Stemmer](https://github.com/har07/Sastrawi)
- [Indonesian Chat Dataset (Kaggle)](https://www.kaggle.com/datasets/jprestiliano/indonesian-chat-dataset)

---

## 🔧 Troubleshooting

### Issue: Model tidak bisa diload
**Solusi**:
```python
from pycaret.classification import load_model
# Pastikan file .pkl ada di working directory
model = load_model("sentiment_pycaret_best")
```

### Issue: Gradio connection error
**Solusi**:
```bash
# Check port availability
lsof -i :7860  # On Mac/Linux
netstat -ano | findstr :7860  # On Windows

# Use different port
python app.py --port 8080
```

### Issue: PyCaret setup error
**Solusi**:
- Upgrade PyCaret: `pip install --upgrade pycaret`
- Check pandas version compatibility
- Ensure text column names match configuration

### Issue: Memory issues saat training
**Solusi**:
- Reduce dataset size untuk testing
- Use `reduce_memory=True` di PyCaret setup
- Increase system memory atau use cloud resources

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Untuk kontribusi:
1. Fork repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📞 Contact & Support

**Project Maintainers**:
- Person 1 (Data Analyst): @allyapasha
- Person 2 (ML Engineer): @eka409

**Issues & Questions**: Create GitHub issue atau contact langsung

---

## 📈 Project Status

- ✅ Data Collection & EDA
- ✅ Preprocessing Pipeline
- ✅ PyCaret Modeling & Tuning
- ✅ Model Evaluation & Documentation
- ⏳ Hugging Face Deployment (in progress)
- ⏳ Final Documentation & Presentation (pending)

**Last Updated**: 2024  
**Version**: 1.0.0

---

**Project**: Sentiment Analysis End-to-End (Collaborative ML)  
**Course**: SD25-32202 - Pemrosesan Bahasa Alami  
**Institution**: Institut Teknologi Sumatera  
