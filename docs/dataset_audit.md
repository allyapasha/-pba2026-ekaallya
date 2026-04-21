# Audit Dataset dan Mapping Label

## Ringkasan Dataset

- Sumber utama training: `C:\Users\ASUS\Downloads\Folder Baru (2)\pba\project-ml\data\raw\sentimentdataset.csv`
- Baris mentah: `732`
- Baris setelah cleaning, drop teks kosong, dan deduplikasi: `707`
- Label mentah unik: `191`

## Distribusi Label 3 Kelas

Distribusi sebelum cleaning:

- `positive`: `480`
- `negative`: `191`
- `neutral`: `61`

Distribusi setelah cleaning:

- `positive`: `459`
- `negative`: `188`
- `neutral`: `60`

## Top 15 Label Mentah

- `Positive`: `45`
- `Joy`: `44`
- `Excitement`: `37`
- `Contentment`: `19`
- `Neutral`: `18`
- `Gratitude`: `18`
- `Curiosity`: `16`
- `Serenity`: `15`
- `Happy`: `14`
- `Nostalgia`: `11`
- `Despair`: `11`
- `Grief`: `9`
- `Awe`: `9`
- `Sad`: `9`
- `Hopeful`: `9`

## Temuan Utama

- Dataset masih timpang ke arah `positive` dengan rasio mayoritas `64.92%`.
- Selisih antara kelas `positive` dan `neutral` setelah cleaning adalah `399` baris.
- Label mentah sangat granular, tetapi sistem deployment tetap harus memetakannya ke 3 kelas output.
- Tidak ada label mentah yang jatuh ke fallback tak dikenal; seluruh label saat ini tertangani oleh mapping.
- Masalah utama ada pada kompresi semantik dan class imbalance, bukan parsing dataset.

## Label Ambigu yang Perlu Diaudit Manual

- `surprise`: bisa positif, negatif, atau netral tergantung konteks
- `nostalgia`: sering bercampur antara positif, sedih, dan reflektif
- `bittersweet`: emosi campuran yang tidak sepenuhnya negatif
- `pressure`: dapat bernuansa stres atau netral tergantung kalimat
- `pensive`: sering reflektif dan tidak selalu negatif
- `sympathy`: lebih dekat ke empati, tidak selalu positif murni

## Dampak ke Model

- Model yang tidak dibalancing akan mudah bias ke `positive` karena kelas ini jauh dominan.
- Kelas `neutral` tetap menjadi kelas tersulit karena jumlah contoh paling kecil dan maknanya paling ambigu.
- Accuracy tidak cukup; baca confusion matrix dan macro F1 sebagai indikator utama.

## Rekomendasi Lanjutan

1. Pertahankan output 3 kelas agar app tetap kompatibel.
2. Audit subset label ambigu sebelum mengubah mapping produksi.
3. Tambah contoh `neutral` jika dataset baru tersedia.
4. Gunakan audit ini sebagai referensi sebelum mengganti model production.
