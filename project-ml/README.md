# Project ML

Folder `project-ml/` dipertahankan sebagai area legacy dan analisis, bukan sebagai pusat production utama.

## Tujuan Folder Ini

Folder ini masih dipakai untuk:

- dataset mentah di `data/raw/`
- dataset hasil preprocessing di `data/processed/`
- notebook EDA dan modeling di `notebooks/`
- wrapper kompatibilitas untuk app lama di `app/`

## Batasan Peran

Folder ini bukan source of truth production untuk:

- training pipeline aktif
- artefak production aktif
- deploy Hugging Face Space

Source of truth production ada di:

- `src/sentiment_project/`
- `pipelines/classic_ml/`
- `artifacts/classic_ml/`
- `apps/local/`
- `apps/hf_space/`

## Isi Penting

- `data/raw/sentimentdataset.csv`
- `data/processed/clean_data.csv`
- `notebooks/01_eda_preprocessing.ipynb`
- `notebooks/02_modeling_pycaret.ipynb`
- `app/app.py`

## Menjalankan Wrapper Lama

```powershell
python .\project-ml\app\app.py
```

Wrapper ini tetap ada agar perintah lama tidak putus, tetapi implementasi app production utama berada di `apps/local/app.py`.
