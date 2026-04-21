# Evaluasi Model Klasik

## Ringkasan
- Model: `LogisticRegression`
- Accuracy: `0.8028`
- F1 weighted: `0.8003`
- F1 macro: `0.7276`
- Distribusi kelas: `{'positive': 459, 'negative': 188, 'neutral': 60}`

## Confusion Matrix
| actual \ predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 26 | 0 | 12 |
| neutral | 2 | 7 | 3 |
| positive | 7 | 4 | 81 |

## Analisis
- Model sebelumnya bias ke `positive` karena distribusi label sangat timpang dan Random Forest cenderung mengikuti kelas mayoritas.
- Logistic Regression dengan `class_weight=balanced` menaikkan recall kelas `negative` dan `neutral` tanpa merusak kontrak probabilitas 3 kelas.
- Kelas `neutral` masih paling sulit karena jumlah contoh sedikit dan semantik beberapa label emosi ambigu.
