# Project ML - Sentiment Analysis

## Deskripsi
Repositori ini disiapkan untuk project **Sentiment Analysis end-to-end** dengan:
- gaya preprocessing dan dokumentasi yang mengacu pada folder `mct-nlp`
- struktur deployment yang mengacu pada folder `deteksi-toksisitas-chat`

Dokumentasi pada file ini difokuskan untuk **Person 1 - Data Analyst (Pre-processing Specialist)**.

## Struktur Folder
```text
project-ml/
├── data/
│   ├── raw/
│   │   └── sentimentdataset.csv
│   └── processed/
│       └── clean_data.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── models/
├── app/
└── README.md
```

## Ringkasan Observasi Awal
### Dataset utama
- Jumlah baris mentah: **732**
- Jumlah kolom mentah: **15**
- Total missing value pada data mentah: **0**
- Jumlah baris setelah preprocessing: **732**
- Jumlah kolom setelah preprocessing: **33**
- Jumlah label sentimen unik: **191**
- Jumlah duplikasi pada kolom teks: **26**
- Baris yang mengandung emoji/non-ASCII pada teks: **18**
- Baris yang mengandung hashtag langsung di kolom teks: **64**

### Top 15 distribusi label sentimen asli
- **Positive**: 45
- **Joy**: 44
- **Excitement**: 37
- **Contentment**: 19
- **Neutral**: 18
- **Gratitude**: 18
- **Curiosity**: 16
- **Serenity**: 15
- **Happy**: 14
- **Nostalgia**: 11
- **Despair**: 11
- **Grief**: 9
- **Awe**: 9
- **Sad**: 9
- **Hopeful**: 9

### Distribusi sentimen hasil pengelompokan 3 kelas
- **positive**: 460
- **negative**: 190
- **neutral**: 82

### Distribusi platform
- **Instagram**: 258
- **Twitter**: 243
- **Facebook**: 231

### Top 10 negara
- **USA**: 188
- **UK**: 143
- **Canada**: 135
- **Australia**: 75
- **India**: 70
- **Brazil**: 17
- **France**: 16
- **Japan**: 15
- **Germany**: 14
- **Italy**: 11

## Standar Preprocessing yang Diadopsi dari `mct-nlp`
Berdasarkan observasi pada modul referensi, pola preprocessing yang diadopsi adalah:
1. mempertahankan kolom teks asli
2. membuat kolom baru bernama `cleaned_text`
3. menggunakan pipeline pembersihan yang eksplisit dan mudah dibaca
4. menghindari preprocessing berlebihan
5. menambahkan contoh before vs after untuk dokumentasi notebook

### Pipeline pembersihan teks
Urutan preprocessing yang digunakan:
1. `lowercase`
2. hapus tag HTML
3. hapus URL
4. ekspansi kontraksi bahasa Inggris
5. ubah hashtag di dalam teks menjadi token biasa
6. hapus mention
7. hapus karakter non-alfabet
8. rapikan whitespace

## Rekayasa Fitur Tambahan
Selain `cleaned_text`, file `clean_data.csv` juga menyimpan fitur tambahan:
- `sentiment_group` sebagai versi agregasi 3 kelas: `positive`, `negative`, `neutral`
- `text_length_chars`
- `text_length_words`
- `cleaned_text_length_chars`
- `cleaned_text_length_words`
- `has_hashtag`
- `hashtag_count`
- `engagement_total`
- kolom hasil scaling z-score untuk fitur numerik utama

## Catatan Pengelompokan Sentimen
Dataset asli memiliki ratusan label emosi granular. Agar lebih mudah digunakan pada tahap modeling, dibuat kolom:
- `sentiment` -> label asli
- `sentiment_group` -> hasil agregasi menjadi 3 kelas

Pendekatan ini dipilih agar:
- analisis EDA tetap mempertahankan label asli
- modeling tahap awal lebih realistis untuk klasifikasi sentimen umum

## Korelasi Fitur Numerik
Berikut korelasi antar fitur numerik utama:

|  | retweets | likes | engagement_total | text_length_chars | text_length_words | cleaned_text_length_chars | cleaned_text_length_words | hour | day | month | year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retweets | 1.0 | 0.998 | 0.999 | 0.202 | 0.197 | 0.201 | 0.2 | 0.197 | 0.009 | 0.073 | -0.04 |
| likes | 0.998 | 1.0 | 1.0 | 0.2 | 0.196 | 0.199 | 0.199 | 0.195 | 0.011 | 0.067 | -0.043 |
| engagement_total | 0.999 | 1.0 | 1.0 | 0.201 | 0.197 | 0.2 | 0.2 | 0.196 | 0.011 | 0.069 | -0.042 |
| text_length_chars | 0.202 | 0.2 | 0.201 | 1.0 | 0.948 | 1.0 | 0.948 | 0.202 | -0.091 | 0.27 | 0.078 |
| text_length_words | 0.197 | 0.196 | 0.197 | 0.948 | 1.0 | 0.949 | 0.996 | 0.225 | -0.083 | 0.246 | 0.041 |
| cleaned_text_length_chars | 0.201 | 0.199 | 0.2 | 1.0 | 0.949 | 1.0 | 0.949 | 0.2 | -0.094 | 0.271 | 0.072 |
| cleaned_text_length_words | 0.2 | 0.199 | 0.2 | 0.948 | 0.996 | 0.949 | 1.0 | 0.226 | -0.085 | 0.248 | 0.042 |
| hour | 0.197 | 0.195 | 0.196 | 0.202 | 0.225 | 0.2 | 0.226 | 1.0 | 0.044 | 0.138 | -0.087 |
| day | 0.009 | 0.011 | 0.011 | -0.091 | -0.083 | -0.094 | -0.085 | 0.044 | 1.0 | -0.136 | 0.022 |
| month | 0.073 | 0.067 | 0.069 | 0.27 | 0.246 | 0.271 | 0.248 | 0.138 | -0.136 | 1.0 | -0.315 |
| year | -0.04 | -0.043 | -0.042 | 0.078 | 0.041 | 0.072 | 0.042 | -0.087 | 0.022 | -0.315 | 1.0 |

## Contoh Sebelum dan Sesudah Preprocessing
| text | cleaned_text | sentiment | sentiment_group |
| --- | --- | --- | --- |
| Exploring the world of digital art. It's never too late to discover new passions. #DigitalArtistry #LateBloomer | exploring the world of digital art it is never too late to discover new passions digitalartistry latebloomer | Curiosity | neutral |
| Feeling inspired after attending a workshop. | feeling inspired after attending a workshop | Positive | positive |
| Eyes wide open in the night, fearful shadows dancing on the walls, the mind a prisoner of imagined horrors. | eyes wide open in the night fearful shadows dancing on the walls the mind a prisoner of imagined horrors | Fearful | negative |
| A soul weathered by the storm of heartbreak, seeking refuge in the calm after. | a soul weathered by the storm of heartbreak seeking refuge in the calm after | Heartbreak | negative |
| Attended a wine tasting event, savoring the richness of flavors that age like fine wine. Cheers to the golden years! #WineTasting #SeniorWineLover | attended a wine tasting event savoring the richness of flavors that age like fine wine cheers to the golden years winetasting seniorwinelover | Joy | positive |
| Despite meticulous training, the swimmer faces disappointment as a split-second miscalculation costs them the lead in a crucial race. | despite meticulous training the swimmer faces disappointment as a split second miscalculation costs them the lead in a crucial race | Miscalculation | negative |
| Excited about the upcoming gaming tournament. | excited about the upcoming gaming tournament | Positive | positive |
| Reflecting on personal growth achieved through life experiences. | reflecting on personal growth achieved through life experiences | Reflection | neutral |

## File yang Dihasilkan
- `data/processed/clean_data.csv`
- `notebooks/01_eda_preprocessing.ipynb`

## Cara Menjalankan Generator
Jalankan perintah berikut dari folder `project-ml`:

```bash
python generate_person1_assets.py
```

## Tugas Person 1 yang Sudah Dicakup
- observasi referensi `mct-nlp`
- observasi dataset `sentimentdataset.csv`
- EDA awal
- preprocessing sesuai standar referensi
- penyimpanan data bersih ke `data/processed/clean_data.csv`
- dokumentasi notebook dan README dalam bahasa Indonesia

## Catatan Lanjutan
Tahap berikutnya yang dapat dikerjakan:
- pembuatan `02_modeling_pycaret.ipynb`
- training model berbasis `cleaned_text` atau kombinasi fitur teks + metadata
- penyusunan app demo mengikuti pola deployment Hugging Face Space
