# Evaluasi Model Klasik

## Ringkasan
- Model: `LogisticRegression`
- Accuracy: `0.8099`
- F1 weighted: `0.7951`
- F1 macro: `0.6857`
- Distribusi kelas: `{'positive': 459, 'negative': 188, 'neutral': 60}`

## Confusion Matrix
| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 24 | 0 | 14 |
| neutral | 1 | 4 | 7 |
| positive | 4 | 1 | 87 |

## Analisis
- Model sebelumnya bias ke `positive` karena distribusi label sangat timpang dan Random Forest cenderung mengikuti kelas mayoritas.
- Logistic Regression dengan `class_weight=balanced` menaikkan recall kelas `negative` dan `neutral` tanpa merusak kontrak probabilitas 3 kelas.
- Kelas `neutral` masih paling sulit karena jumlah contoh sedikit dan semantik beberapa label emosi ambigu.
