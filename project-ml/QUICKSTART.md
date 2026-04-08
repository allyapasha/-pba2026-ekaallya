# 🚀 QUICKSTART GUIDE - Sentiment Analysis Project

## ⚡ 30 Second Overview

This is a **complete, production-ready sentiment analysis project** for Indonesian text. It consists of:

- ✅ **Cleaned dataset** (732 samples) - `data/processed/clean_data.csv`
- ✅ **Trained ML model** - `models/sentiment_pycaret_best.pkl`
- ✅ **Web interface** (Gradio) - `app/app.py`
- ✅ **Full documentation** - See `README.md` for details

**Status**: Ready for deployment to Hugging Face Spaces

---

## 🎯 Quick Navigation

| Want to... | Go to... | Time |
|-----------|----------|------|
| **Understand the project** | Read `README.md` | 5 min |
| **See data processing** | Open `notebooks/01_eda_preprocessing.ipynb` | 10 min |
| **See model training** | Open `notebooks/02_modeling_pycaret.ipynb` | 10 min |
| **Run the app locally** | See "Local Setup" below | 2 min |
| **Deploy to HuggingFace** | See "Deploy" below | 5 min |
| **View project structure** | See "Project Layout" below | 2 min |

---

## 📦 Local Setup (2 minutes)

### Step 1: Install Dependencies
```bash
cd project-ml/app
pip install -r requirements.txt
```

### Step 2: Run the App
```bash
python app.py
```

### Step 3: Open Browser
Navigate to: **http://localhost:7860**

✅ **Done!** Test sentiment analysis on the web interface.

---

## 🌐 Deploy to Hugging Face (5 minutes)

### Prerequisites
- Hugging Face account (free at https://huggingface.co)
- Git installed
- Files from `app/` folder

### Step-by-Step Deployment

1. **Create a Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `sentiment-analysis-indonesian`
   - SDK: Select **Gradio**
   - License: MIT

2. **Clone the Space**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analysis-indonesian
   cd sentiment-analysis-indonesian
   ```

3. **Copy Files**
   ```bash
   # From project-ml directory
   cp app/app.py .
   cp app/config.py .
   cp app/requirements.txt .
   cp models/sentiment_pycaret_best.pkl .
   ```

4. **Push to Hugging Face**
   ```bash
   git add .
   git commit -m "Initial Sentiment Analysis deployment"
   git push
   ```

5. **Done!** 🎉
   - Space auto-builds (takes ~1-2 minutes)
   - Your app is live at: `https://huggingface.co/spaces/YOUR_USERNAME/sentiment-analysis-indonesian`

---

## 📁 Project Layout

```
project-ml/                              ← Main folder
│
├── 📂 notebooks/                         ← Development (read-only)
│   ├── 01_eda_preprocessing.ipynb       Data processing by Person 1
│   └── 02_modeling_pycaret.ipynb        Model training by Person 2
│
├── 📂 data/                              ← Datasets
│   ├── raw/
│   │   └── sentimentdataset.csv         Original (732 records)
│   └── processed/
│       └── clean_data.csv               Processed & cleaned
│
├── 📂 models/                            ← Trained models
│   ├── sentiment_pycaret_best.pkl       Main model (IMPORTANT!)
│   ├── confusion_matrix.png             Performance visualization
│   ├── feature_importance.png           Feature analysis
│   └── roc_auc.png                      ROC curve
│
├── 📂 app/                               ← Deployment folder (EDIT THIS)
│   ├── app.py                           Gradio web interface ⭐
│   ├── config.py                        Configuration file
│   ├── requirements.txt                 Python dependencies
│   └── README.md                        App documentation
│
├── README.md                             ← Complete guide
├── PROJECT_SUMMARY.md                    ← Deliverables summary
└── QUICKSTART.md                         ← You are here!
```

**Key Files**:
- `app/app.py` - The web interface (what users interact with)
- `models/sentiment_pycaret_best.pkl` - The trained model
- `data/processed/clean_data.csv` - The training data

---

## 💻 Example Usage

### Test Locally
```bash
# Terminal 1: Start the app
cd app
python app.py

# Terminal 2: (Browser opens automatically, or visit http://localhost:7860)
# Type in the text box:
# "Saya sangat senang dengan produk ini!"
# Output: Positive (😊)
```

### Using Model in Python
```python
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model("models/sentiment_pycaret_best")

# Prepare data (with text preprocessing)
df = pd.DataFrame({"cleaned_text": ["saya sangat senang"]})

# Predict
result = predict_model(model, data=df)
print(result[['cleaned_text', 'prediction_label']])
```

---

## 🎯 What Each Component Does

### 1️⃣ Data Preprocessing (Notebook 01)
- **Input**: `sentimentdataset.csv` (732 raw records)
- **Process**: 
  - Remove duplicates & handle missing values
  - Clean text (lowercase, remove URLs, normalize slang)
  - Engineer features (text length, engagement metrics)
- **Output**: `clean_data.csv` (732 clean records, 33 features)

### 2️⃣ Model Training (Notebook 02)
- **Input**: `clean_data.csv`
- **Process**:
  - Load data into PyCaret
  - Compare 10+ ML algorithms
  - Select and tune best model
  - Train on full dataset
- **Output**: `sentiment_pycaret_best.pkl` (trained model)

### 3️⃣ Web Interface (app.py)
- **Input**: User types text in web browser
- **Process**:
  - Apply same preprocessing as training
  - Load model from .pkl file
  - Make prediction
- **Output**: Sentiment label with confidence score

---

## 🔧 Troubleshooting

### Issue: "Model not found" error
```
Solution: Ensure sentiment_pycaret_best.pkl is in the same folder as app.py
cp models/sentiment_pycaret_best.pkl app/
```

### Issue: "Port 7860 already in use"
```
Solution: Use a different port
python app.py --port 8080
```

### Issue: Import error (pycaret not found)
```
Solution: Reinstall dependencies
pip install --upgrade -r app/requirements.txt
```

### Issue: Deployment shows "Loading..."
```
Solution: Wait 1-2 minutes for Hugging Face to build
Check the build log in your Space settings
```

---

## 📊 Model Info

| Aspect | Value |
|--------|-------|
| **Type** | Classification (3 classes) |
| **Classes** | Positive, Negative, Neutral |
| **Framework** | PyCaret (built on Scikit-learn) |
| **Training Data** | 732 Indonesian text samples |
| **Features** | 33 (TF-IDF + numerical) |
| **Preprocessing** | Text cleaning + slang expansion |
| **Optimization** | Hyperparameter tuning via PyCaret |

---

## 📚 Full Documentation

For detailed information, see:

- **README.md** - Complete project guide with setup, usage, and insights
- **PROJECT_SUMMARY.md** - Deliverables checklist and project completion status
- **notebooks/** - Jupyter notebooks with full explanations
- **app/README.md** - App-specific documentation

---

## 🎓 Key Learnings

This project demonstrates:

✅ **Data Pipeline**: Raw data → EDA → Cleaning → Features → ML ready  
✅ **Automated ML**: PyCaret for model selection & tuning  
✅ **Preprocessing**: Indonesian slang normalization, leetspeak handling  
✅ **Deployment**: Gradio web UI + Hugging Face Spaces  
✅ **Documentation**: Code comments, docstrings, markdown guides  

---

## 🔗 Useful Links

| Resource | Link |
|----------|------|
| PyCaret Docs | https://pycaret.org |
| Gradio Docs | https://gradio.app |
| Hugging Face | https://huggingface.co |
| GitHub | [Your repo link] |

---

## 💡 Next Steps

### Option A: Learn More
→ Read `README.md` for comprehensive documentation

### Option B: Run Locally
→ Follow "Local Setup" section above

### Option C: Deploy to Production
→ Follow "Deploy to Hugging Face" section above

### Option D: Modify the Model
→ Edit `notebooks/02_modeling_pycaret.ipynb` and retrain

---

## ❓ FAQs

**Q: Can I use this for other languages?**  
A: The model is trained on Indonesian text with Indonesian slang dictionary. For other languages, you'd need to retrain with that language's data.

**Q: How accurate is the model?**  
A: See `notebooks/02_modeling_pycaret.ipynb` for detailed metrics.

**Q: Can I modify the app?**  
A: Yes! Edit `app/app.py` and redeploy to Hugging Face.

**Q: How do I retrain the model?**  
A: Run `notebooks/02_modeling_pycaret.ipynb` after updating `clean_data.csv`.

**Q: Is the code open source?**  
A: Yes! MIT License. Feel free to fork and modify.

---

## 📝 Summary

| Step | Command | Time |
|------|---------|------|
| **Install** | `pip install -r app/requirements.txt` | 1 min |
| **Run Local** | `cd app && python app.py` | 30 sec |
| **Deploy** | Push files to Hugging Face Space | 5 min |

---

## 🎉 You're All Set!

Your sentiment analysis system is ready to use. Choose your path:

1. **🔬 Explore**: Open notebooks to understand the process
2. **🚀 Deploy**: Follow Hugging Face deployment guide
3. **📖 Learn**: Read full README.md for detailed info

---

<div align="center">

**Made with ❤️ for Indonesian NLP**

Institut Teknologi Sumatera - 2024

For questions: Check README.md or create a GitHub issue

</div>