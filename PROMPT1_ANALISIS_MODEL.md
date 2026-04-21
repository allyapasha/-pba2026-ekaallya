# Analisis Prompt 1

## Ringkasan Teknis Model

Model pada repo ini bukan deep learning. Pipeline training utama di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:24) memakai `TfidfVectorizer`, `StandardScaler`, `LabelEncoder`, dan `RandomForestClassifier`, terlihat dari impor dan tahap training di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:443) dan [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:483). Ringkasan artefak model di [project-ml/models/MODEL_SUMMARY.txt](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/models/MODEL_SUMMARY.txt:1) juga menyebut algoritma `Random Forest` dengan artefak `sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`, dan `scaler.pkl`.

Output target model adalah klasifikasi 3 kelas: `positive`, `negative`, dan `neutral`. Di pipeline training, label target dibentuk melalui `sentiment_group` pada [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:393), lalu di-encode untuk training pada [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:470).

## Alur Preprocessing

Dataset dibaca dari `project-ml/data/raw/sentimentdataset.csv` menggunakan `pd.read_csv(..., sep=";")` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:271). Ini penting karena file `sentimentdataset.csv` memang berformat delimiter `;`, dengan kolom seperti `Text`, `Sentiment`, `Timestamp`, `Hashtags`, `Retweets`, dan `Likes`.

Preprocessing teks dilakukan oleh `clean_text()` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:350). Urutannya:

- teks diubah ke huruf kecil
- URL dihapus
- mention `@user` dihapus
- leetspeak dinormalisasi
- slang diperluas ke bentuk baku
- karakter non-huruf dibuang
- spasi dirapikan

Setelah itu:

- `Text` dibersihkan menjadi `cleaned_text` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:377)
- label `Sentiment` dibersihkan dengan `strip()`
- label emosi asli dipetakan ke 3 kelas lewat `map_sentiment_label()` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:364)
- hasil akhirnya disimpan sebagai `sentiment_group` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:393)

## Struktur Fitur

Fitur model terdiri dari dua bagian:

- Fitur teks: `cleaned_text` diubah menjadi vektor TF-IDF memakai `TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8, ngram_range=(1, 2))` pada [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:443).
- Fitur numerik: `text_length_words`, `engagement_total`, dan `hashtag_count`, dibentuk di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:401) lalu diskalakan dengan `StandardScaler` pada [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:461).

Kedua komponen digabung menjadi satu matriks fitur dengan `np.hstack(...)` di [run_simple_pipeline.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/run_simple_pipeline.py:465), lalu dipakai untuk training Random Forest.

## Kesesuaian Input Model Dengan Dataset

Secara struktur, dataset `C:\\Users\\ASUS\\Downloads\\Folder Baru (2)\\pba\\sentimentdataset.csv` cocok untuk pipeline ini karena memiliki:

- `Text` untuk sumber fitur TF-IDF
- `Sentiment` untuk target klasifikasi
- `Retweets` dan `Likes` untuk `engagement_total`
- `Hashtags` untuk `hashtag_count`

Namun ada beberapa hal penting:

- Dataset memakai delimiter `;`, jadi pembacaan default CSV tanpa `sep=";"` akan salah.
- Kolom `Sentiment` pada dataset berisi banyak label emosi, bukan langsung 3 kelas. Karena itu perlu mapping ke `positive`, `negative`, `neutral`.
- Dua kolom indeks tambahan (`Unnamed: 0.1`, `Unnamed: 0`) tidak dipakai model dan aman diabaikan.
- Teks dataset mayoritas berbahasa Inggris, sedangkan preprocessing slang di app lebih diarahkan ke bahasa Indonesia. Ini tidak merusak pipeline, tetapi manfaat normalisasi slang Indonesia menjadi lebih kecil untuk dataset ini.

## Bentuk Output Prediksi Di Aplikasi

Aplikasi inferensi di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:146) memuat empat artefak: model, vectorizer, label encoder, dan scaler. Fungsi `predict_sentiment()` di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:203) menjalankan preprocessing yang sejalan dengan training:

- membersihkan teks dengan `clean_text()` di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:179)
- membentuk fitur inferensi lewat `build_feature_frame()` di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:192)
- membuat fitur TF-IDF dan fitur numerik
- menggabungkan semuanya ke `model_input`
- memanggil `MODEL.predict_proba(...)` di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:219)

Output aplikasi bukan satu label tunggal, melainkan dictionary probabilitas untuk tiap kelas, lalu ditampilkan melalui `gr.Label(num_top_classes=3)` di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:245). Jadi bentuk output inferensi adalah skor probabilitas `positive`, `negative`, dan `neutral`.

## Implikasi Terhadap Inferensi

- Karena training memakai 3 fitur numerik (`text_length_words`, `engagement_total`, `hashtag_count`), app juga harus membentuk 3 fitur yang sama. Ini sudah konsisten di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:192).
- Pada inferensi teks tunggal, `engagement_total` dan `hashtag_count` saat ini diisi `0` di app. Artinya prediksi aplikasi terutama digerakkan oleh fitur TF-IDF dan panjang teks, bukan metadata sosial yang ada di dataset training.
- Karena dataset asli multi-emotion dipaksa menjadi 3 kelas, sebagian nuansa emosi hilang. Model akhirnya adalah sentiment classifier 3 kelas, bukan emotion classifier.
- Ada potensi mismatch domain: dataset banyak teks bahasa Inggris, sementara app dideskripsikan sebagai analisis sentimen bahasa Indonesia di [project-ml/app/app.py](/C:/Users/ASUS/Downloads/Folder%20Baru%20(2)/pba/project-ml/app/app.py:233). Secara teknis aplikasi tetap jalan, tetapi klaim domain bahasa Indonesia perlu dianggap terbatas jika model dilatih dominan dari dataset ini.

## Kesimpulan

Pipeline saat ini adalah pipeline machine learning klasik, bukan deep learning: `clean_text` -> `TF-IDF + fitur numerik` -> `StandardScaler` -> `RandomForestClassifier` -> output probabilitas 3 kelas. Dataset `sentimentdataset.csv` kompatibel setelah pembacaan delimiter `;` dan normalisasi label dilakukan. Inferensi di app konsisten dengan artefak training, tetapi outputnya harus dipahami sebagai sentimen 3 kelas yang menyederhanakan label emosi asli dataset.
