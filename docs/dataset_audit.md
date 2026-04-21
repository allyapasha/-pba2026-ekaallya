# Audit Dataset dan Mapping Label

## Ringkasan Dataset

- Sumber utama training: `project-ml/data/raw/sentimentdataset.csv`
- Salinan tambahan: `sentimentdataset.csv`
- Baris mentah: `732`
- Baris setelah cleaning, drop teks kosong, dan deduplikasi: `707`
- Label mentah unik: `191`

## Distribusi Label

Distribusi sebelum cleaning:

- `positive`: `480`
- `negative`: `191`
- `neutral`: `61`

Distribusi setelah cleaning:

- `positive`: `459`
- `negative`: `188`
- `neutral`: `60`

## Temuan Utama

- Dataset sangat timpang ke arah `positive`.
- Label mentah sangat granular, tetapi dipaksa ke 3 kelas output.
- Tidak ada label mentah yang jatuh ke `neutral` karena tidak dikenali. Semua label saat ini sudah tertangani oleh mapping.
- Masalah utama bukan label tak dikenal, tetapi kompresi semantik yang terlalu agresif.

## Risiko Mapping Saat Ini

Beberapa label yang patut dikaji ulang pada iterasi berikutnya:

- `surprise`
  - Saat ini dipetakan ke `positive`, padahal secara semantik bisa positif, negatif, atau netral tergantung konteks.
- `nostalgia`
  - Saat ini dipetakan ke `positive`, padahal sering bercampur dengan kesedihan atau refleksi netral.
- `bittersweet`
  - Saat ini dipetakan ke `negative`, padahal label ini campuran.
- `challenge`, `pressure`, `pensive`
  - Saat ini dipetakan ke `negative`, tetapi sebagian konteks bisa netral atau reflektif.
- `sympathy`
  - Saat ini dipetakan ke `positive`, walau pada banyak kasus lebih dekat ke empati netral.

## Dampak ke Model

- Model lama cenderung memprediksi `positive` karena kelas mayoritas jauh lebih banyak.
- Kelas `neutral` menjadi paling sulit karena datanya paling sedikit dan semantik labelnya paling ambigu.
- Evaluasi model harus dibaca bersama distribusi label, bukan hanya accuracy.

## Rekomendasi

1. Pertahankan mapping 3 kelas untuk kompatibilitas app.
2. Audit manual subset label ambigu pada iterasi berikutnya.
3. Tambahkan data `neutral` yang lebih representatif jika memungkinkan.
4. Gunakan metrik macro F1 dan confusion matrix sebagai indikator utama, bukan accuracy saja.

