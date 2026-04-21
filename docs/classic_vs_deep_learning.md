# Perbandingan Model Klasik vs Baseline Neural

## Model Klasik Production

- Algoritma: `LogisticRegression`
- Fitur: `TF-IDF` + fitur numerik sederhana
- Accuracy: `0.8099`
- F1 weighted: `0.7951`
- F1 macro: `0.6857`

Kelebihan:

- ringan
- cepat di-train
- mudah di-pack ke Hugging Face Space
- mudah dipahami dan dirawat

Kekurangan:

- masih sensitif terhadap distribusi label yang timpang
- perlu penyesuaian inferensi agar contoh Indonesia sederhana tidak terlalu bias

## Baseline Neural Eksperimen

- Algoritma: `MLPClassifier`
- Fitur: `TF-IDF` + fitur numerik sederhana
- Accuracy: `0.8451`
- F1 weighted: `0.8304`
- F1 macro: `0.7263`

Kelebihan:

- performa evaluasi lebih tinggi
- memberi titik awal eksperimen neural yang murah

Kekurangan:

- belum sequence-aware seperti LSTM atau transformer
- environment lokal saat ini belum menyediakan PyTorch atau TensorFlow
- belum diaudit untuk jalur deployment production
- belum diintegrasikan ke app dan Space agar tidak menambah kompleksitas deploy

## Kapan Dipakai

- Pakai model klasik untuk production lokal dan Hugging Face Space.
- Pakai baseline neural untuk eksperimen lanjutan dan pembanding performa.

## Keputusan Prompt 2

Default production tetap model klasik, walaupun baseline neural lebih tinggi, karena target Prompt 2 adalah repo yang stabil, rapi, dan siap deploy. Jalur neural dipisahkan agar eksperimen bisa lanjut tanpa mengganggu app yang sudah berjalan.

## Catatan Terminologi

Folder `deep_learning/` dipertahankan untuk konsistensi struktur eksperimen, tetapi baseline yang aktif saat ini adalah neural baseline ringan berbasis `MLPClassifier`. Ini adalah langkah transisi yang jujur secara teknis sampai environment menyediakan dependensi deep learning penuh.

