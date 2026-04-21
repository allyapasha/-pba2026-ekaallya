# Prompt 2 Handoff

## Status Setelah Prompt 1

Sistem lokal sudah dirapikan dan berjalan pada level berikut:

- pipeline training utama: `run_simple_pipeline.py`
- modul bersama: `sentiment_system.py`
- app inferensi lokal: `project-ml/app/app.py`
- smoke test lokal: `validate_local_system.py`
- artefak model terbaru sudah direfresh di `project-ml/models/`
- dataset hasil preprocessing terbaru sudah ada di `project-ml/data/processed/clean_data.csv`

## Jalur Kerja Yang Sudah Stabil

1. Training lokal

```powershell
python run_simple_pipeline.py
```

2. Validasi sistem lokal

```powershell
python validate_local_system.py
```

3. Menjalankan app lokal

```powershell
pip install -r .\project-ml\app\requirements.txt
python .\project-ml\app\app.py
```

## Hal Penting Untuk Prompt 2

- Gunakan `sentiment_system.py` sebagai sumber tunggal untuk:
  - pembacaan dataset
  - pembersihan teks
  - mapping label
  - fitur inferensi
- Jangan duplikasi lagi logika preprocessing di file lain.
- Dataset ada dalam dua salinan dengan delimiter berbeda:
  - `sentimentdataset.csv` di root: delimiter `;`
  - `project-ml/data/raw/sentimentdataset.csv`: delimiter `,`
- Loader di `sentiment_system.py` sudah dibuat tahan terhadap keduanya.

## Fokus Yang Pantas Dilanjutkan Di Prompt 2

- evaluasi kualitas mapping label 3 kelas
- penanganan class imbalance
- evaluasi kualitas prediksi yang masih bias ke `positive`
- perbaikan dataset atau strategi model tanpa merusak kontrak output app
