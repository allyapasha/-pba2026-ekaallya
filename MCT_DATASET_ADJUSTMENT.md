# MCT Dataset Adjustment

## Ringkasan Temuan

- Folder `mct-nlp` tidak berisi source model yang bisa dipakai; implementasi model yang aktif ada di root repo melalui `run_simple_pipeline.py`, `run_pipeline.py`, dan `project-ml/app/app.py`.
- Repo ini bukan deep learning. Model yang benar-benar dipakai saat ini adalah:
  - `run_simple_pipeline.py`: `TfidfVectorizer` + `RandomForestClassifier`
  - `run_pipeline.py`: PyCaret AutoML untuk klasifikasi
- Dataset `sentimentdataset.csv` memakai delimiter `;`, bukan `,`.
- Dataset memiliki 732 baris dan label multi-emotion, bukan hanya 3 label `positive/negative/neutral`.

## Penyesuaian yang Dibutuhkan

Supaya repo tetap konsisten dengan output aplikasi 3 kelas, label emosi perlu diagregasi ke:

- `positive`
- `negative`
- `neutral`

Distribusi hasil mapping setelah penyesuaian:

- `positive`: 480
- `negative`: 191
- `neutral`: 61

## Penyesuaian yang Sudah Diterapkan

- `run_simple_pipeline.py`
  - pembacaan CSV diubah ke `sep=";"`
  - nama kolom dinormalisasi dengan `strip()`
  - label `Sentiment` dibersihkan dengan `strip()`
  - label emosi multi-class dipetakan ke 3 kelas output
- `run_pipeline.py`
  - pembacaan CSV diubah ke `sep=";"`
  - nama kolom dinormalisasi dengan `strip()`
  - label `Sentiment` dibersihkan dengan `strip()`
  - label emosi multi-class dipetakan ke 3 kelas output

## 2 Prompt Untuk 2 Orang

### Prompt 1

Pahami pipeline model pada repo ini dengan fokus pada `run_simple_pipeline.py`, `project-ml/models/`, dan `project-ml/app/app.py`. Pastikan analisis menjelaskan bahwa model yang digunakan saat ini bukan deep learning, melainkan TF-IDF + Random Forest untuk klasifikasi 3 kelas (`positive`, `negative`, `neutral`). Lalu cek kesesuaian input model dengan dataset `C:\\Users\\ASUS\\Downloads\\Folder Baru (2)\\pba\\sentimentdataset.csv`, terutama format kolom, delimiter `;`, preprocessing teks, fitur numerik, dan bentuk output prediksi di aplikasi. Hasil akhir yang diminta adalah ringkasan teknis model, alur preprocessing, struktur fitur, serta implikasi terhadap inferensi.

### Prompt 2

Pahami dataset `C:\\Users\\ASUS\\Downloads\\Folder Baru (2)\\pba\\sentimentdataset.csv` lalu sesuaikan pipeline training agar kompatibel dengan dataset tersebut tanpa mengubah target output aplikasi yang tetap 3 kelas. Fokus pada audit label `Sentiment`, normalisasi kolom, penanganan delimiter `;`, pembersihan whitespace, dan pemetaan label emosi yang banyak menjadi `positive`, `negative`, dan `neutral`. Hasil akhir yang diminta adalah daftar penyesuaian kode yang perlu/ sudah dilakukan, distribusi label setelah mapping, risiko class imbalance, dan rekomendasi validasi sebelum retraining model.
