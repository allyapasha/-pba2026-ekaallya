# 📑 PROJECT INDEX - Sentiment Analysis End-to-End

**Quick Navigation Guide for the Complete Sentiment Analysis Project**

---

## 🚀 START HERE

### First Time? Read This First
1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ - 5-minute setup guide
2. **[README.md](README.md)** - Comprehensive documentation
3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Deliverables & achievements

---

## 📂 DIRECTORY STRUCTURE

### `notebooks/` - Development & Analysis
| File | Purpose | Author | Status |
|------|---------|--------|--------|
| **01_eda_preprocessing.ipynb** | EDA, data cleaning, feature engineering | Person 1 | ✅ Complete |
| **02_modeling_pycaret.ipynb** | Model training, tuning, evaluation | Person 2 | ✅ Complete |

**How to Use**:
```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
jupyter notebook notebooks/02_modeling_pycaret.ipynb
```

---

### `data/` - Datasets
| File | Records | Columns | Purpose |
|------|---------|---------|---------|
| `raw/sentimentdataset.csv` | 732 | 14 | Original dataset |
| `processed/clean_data.csv` | 732 | 33 | Preprocessed & features |

**Data Flow**:
```
sentimentdataset.csv (Raw)
         ↓
[01_eda_preprocessing.ipynb] (Person 1)
         ↓
clean_data.csv (Processed)
         ↓
[02_modeling_pycaret.ipynb] (Person 2)
         ↓
sentiment_pycaret_best.pkl (Model)
```

---

### `models/` - Trained Models & Artifacts
| File | Type | Purpose |
|------|------|---------|
| **sentiment_pycaret_best.pkl** | Model | 🏆 Main trained model |
| sentiment_pycaret_best.zip | Compressed | Compressed model backup |
| confusion_matrix.png | Visualization | Test set confusion matrix |
| feature_importance.png | Visualization | Top predictive features |
| roc_auc.png | Visualization | ROC AUC curve |
| MODEL_SUMMARY.txt | Documentation | Model performance metrics |

**How to Load**:
```python
from pycaret.classification import load_model
model = load_model("models/sentiment_pycaret_best")
```

---

### `app/` - Web Application (Deployment)
| File | Purpose | Role |
|------|---------|------|
| **app.py** | 🌐 Gradio web interface | Main app file |
| **config.py** | ⚙️ Configuration & constants | Settings module |
| **requirements.txt** | 📦 Python dependencies | Package list |
| README.md | 📖 App documentation | App-specific guide |

**How to Run**:
```bash
cd app
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

---

## 📚 DOCUMENTATION FILES

### Main Documentation
| File | Content | Read Time |
|------|---------|-----------|
| **README.md** | Complete project guide, setup, usage, deployment | 15 min |
| **PROJECT_SUMMARY.md** | Deliverables checklist, key metrics, status | 10 min |
| **QUICKSTART.md** | Fast setup guide, 3 options to run | 5 min |
| **INDEX.md** | This file - navigation guide | 3 min |

---

## 🎯 QUICK START PATHS

### Path 1: View & Understand (No Installation)
1. Open `README.md` - Read overview
2. Open `notebooks/01_eda_preprocessing.ipynb` - View analysis
3. Open `notebooks/02_modeling_pycaret.ipynb` - View training

**Time**: 15 minutes

---

### Path 2: Run Locally (Installation Required)
```bash
# 1. Install dependencies
cd app
pip install -r requirements.txt

# 2. Run app
python app.py

# 3. Open browser
# http://localhost:7860
```

**Time**: 5 minutes

---

### Path 3: Deploy to Production
```bash
# 1. Create space at huggingface.co/spaces
# 2. Clone space repo
# 3. Copy files from app/ and models/
# 4. Push to Hugging Face
git push
```

**Time**: 10 minutes + auto-build (2 min)

---

## 🔍 FIND INFORMATION BY TOPIC

### Project Overview
- **What is this project?** → [README.md - Deskripsi Proyek](README.md#-deskripsi-proyek)
- **Who worked on it?** → [README.md - Anggota Tim](README.md#-anggota-tim)
- **Project status?** → [PROJECT_SUMMARY.md - Status](PROJECT_SUMMARY.md#-project-completion-status)

### Data & Preprocessing
- **Dataset info?** → [README.md - Dataset](README.md#-dataset)
- **Data cleaning steps?** → [notebooks/01_eda_preprocessing.ipynb](notebooks/01_eda_preprocessing.ipynb)
- **Features created?** → [PROJECT_SUMMARY.md - Data Stats](PROJECT_SUMMARY.md#dataset)
- **Slang dictionary?** → [app/config.py - SLANG_DICT](app/config.py#💬-kamus-slang-indonesia)

### Model & Training
- **How was model trained?** → [notebooks/02_modeling_pycaret.ipynb](notebooks/02_modeling_pycaret.ipynb)
- **Model performance?** → [PROJECT_SUMMARY.md - Model Performance](PROJECT_SUMMARY.md#🎯-key-metrics--results)
- **Which algorithms?** → [README.md - Model Comparison](README.md#-model--performa)
- **Feature importance?** → [models/feature_importance.png](models/feature_importance.png)

### Deployment & Web
- **How to run app?** → [QUICKSTART.md - Option 2](QUICKSTART.md#-option-2-run-web-app-local)
- **How to deploy?** → [QUICKSTART.md - Option 3](QUICKSTART.md#-option-3-deploy-to-hugging-face-spaces)
- **App code?** → [app/app.py](app/app.py)
- **App configuration?** → [app/config.py](app/config.py)

### Troubleshooting
- **Common issues?** → [QUICKSTART.md - Troubleshooting](QUICKSTART.md#-troubleshooting)
- **Port already in use?** → [QUICKSTART.md - Port Error](QUICKSTART.md#problem-port-7860-already-in-use)
- **Model not found?** → [QUICKSTART.md - Model Error](QUICKSTART.md#issue-model-not-found-error)

---

## 📊 PROJECT STATISTICS

| Metric | Value |
|--------|-------|
| **Total Files** | 12 main files |
| **Lines of Code** | 3,000+ lines |
| **Documentation** | 4 markdown files |
| **Notebooks** | 2 Jupyter notebooks |
| **Data Records** | 732 samples |
| **Features** | 33 engineered |
| **Classes** | 3 (positive, negative, neutral) |
| **Model Type** | PyCaret AutoML |
| **Framework** | Python 3.10+ |

---

## 🎓 LEARNING OBJECTIVES MET

✅ **Data Science**: EDA, preprocessing, feature engineering  
✅ **Machine Learning**: Model selection, tuning, evaluation  
✅ **NLP**: Text preprocessing, slang normalization, TF-IDF  
✅ **Web Development**: Gradio interface, web deployment  
✅ **DevOps**: Hugging Face Spaces, Docker-ready  
✅ **Software Engineering**: Code quality, documentation, standards  

---

## 🚀 DEPLOYMENT CHECKLIST

- [x] Model trained and saved
- [x] Web interface created (Gradio)
- [x] Configuration management (config.py)
- [x] Requirements.txt prepared
- [x] Local testing completed
- [x] Documentation written
- [ ] Deploy to Hugging Face Spaces (manual step)

**To Deploy**:
1. Go to https://huggingface.co/spaces
2. Create new Space (Gradio)
3. Copy files from `app/` and `models/`
4. Push to Space repo
5. ✅ Live!

---

## 💡 KEY FEATURES

### Data Processing
- 🔤 Leetspeak normalization (4nj1n9 → anjing)
- 💬 Indonesian slang expansion (150+ mappings)
- 🧹 Text cleaning pipeline (7 steps)
- 📊 Feature engineering (19 new features)

### Machine Learning
- 🤖 PyCaret AutoML (10+ algorithms tested)
- 🎯 Hyperparameter tuning (RandomizedSearchCV)
- 📈 Performance visualization (confusion matrix, ROC AUC)
- ✅ Stratified cross-validation (5-fold)

### Deployment
- 🌐 Gradio web interface
- ☁️ Hugging Face Spaces ready
- 🔒 Model versioning (.pkl)
- 📚 Comprehensive documentation

---

## 📞 SUPPORT & HELP

### Getting Help
1. **Quick issue?** → Check [QUICKSTART.md - Troubleshooting](QUICKSTART.md#-troubleshooting)
2. **Technical question?** → See [README.md - Troubleshooting](README.md#-troubleshooting)
3. **Need to understand code?** → Read notebook comments and docstrings
4. **Want to modify?** → Edit `app/app.py` or notebooks and redeploy

### Common Questions
- **How accurate is the model?** → See [notebooks/02_modeling_pycaret.ipynb](notebooks/02_modeling_pycaret.ipynb) output
- **Can I use this for other languages?** → Retrain with different language data
- **How do I add more data?** → Update `data/raw/`, rerun preprocessing notebook
- **Can I modify the interface?** → Yes! Edit `app/app.py` and redeploy

---

## 🎯 NEXT STEPS

**Choose Your Path**:

### 🔬 Learn & Explore
1. Open `README.md` - Understand the project
2. Open `notebooks/01_eda_preprocessing.ipynb` - See data analysis
3. Open `notebooks/02_modeling_pycaret.ipynb` - See model training
4. Examine `app/app.py` - Understand deployment

### 🚀 Run Locally
1. Follow [QUICKSTART.md - Local Setup](QUICKSTART.md#-local-setup-2-minutes)
2. Test predictions at http://localhost:7860
3. Explore app interface and examples

### 🌐 Deploy to Production
1. Follow [QUICKSTART.md - Deploy](QUICKSTART.md#-deploy-to-hugging-face-5-minutes)
2. Share public link with others
3. Monitor usage and performance

### 📈 Extend the Project
1. Collect more data
2. Train with additional features
3. Try different algorithms
4. Deploy improved version

---

## 📝 FILE QUICK REFERENCE

```
project-ml/
├── README.md                    ← Read this first for complete info
├── QUICKSTART.md               ← Follow this for fast setup
├── PROJECT_SUMMARY.md          ← See deliverables & achievements
├── INDEX.md                    ← You are here!
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb    (Person 1: Data Analysis)
│   └── 02_modeling_pycaret.ipynb     (Person 2: Model Training)
│
├── data/
│   ├── raw/sentimentdataset.csv      (Original: 732 × 14)
│   └── processed/clean_data.csv      (Cleaned: 732 × 33)
│
├── models/
│   ├── sentiment_pycaret_best.pkl    ⭐ Main model
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_auc.png
│
└── app/
    ├── app.py                 ⭐ Web interface
    ├── config.py              Configuration
    ├── requirements.txt        Dependencies
    └── README.md               App docs
```

---

## ⏱️ TIME ESTIMATES

| Task | Time |
|------|------|
| Read QUICKSTART | 5 min |
| Read README | 15 min |
| Run app locally | 5 min |
| Deploy to HF | 10 min |
| Total project understanding | 30 min |

---

## 🎉 SUMMARY

You now have a **complete, production-ready sentiment analysis system** for Indonesian text with:

✅ Clean, preprocessed dataset (732 samples)  
✅ Trained ML model (PyCaret AutoML)  
✅ Web interface (Gradio)  
✅ Deployment ready (Hugging Face)  
✅ Full documentation  
✅ Example code  
✅ Troubleshooting guide  

**Choose your starting point from the paths above and begin!**

---

<div align="center">

**Made with ❤️ for Easy Project Navigation**

Institut Teknologi Sumatera - SD25-32202 NLP Course - 2024

[README](README.md) | [QUICKSTART](QUICKSTART.md) | [PROJECT_SUMMARY](PROJECT_SUMMARY.md)

</div>