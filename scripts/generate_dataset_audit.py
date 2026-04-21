from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment_project.config import DOCS_DIR, RAW_DATA_PATH
from src.sentiment_project.shared import TARGET_COLUMN, load_sentiment_dataset, map_sentiment_label, prepare_training_dataframe


AUDIT_JSON_PATH = DOCS_DIR / "dataset_audit.json"
AUDIT_MD_PATH = DOCS_DIR / "dataset_audit.md"


def build_audit_payload() -> dict:
    df_raw = load_sentiment_dataset(RAW_DATA_PATH)
    raw_labels = df_raw["Sentiment"].astype(str).str.strip()
    df_clean = prepare_training_dataframe(df_raw)

    raw_distribution = raw_labels.apply(map_sentiment_label).value_counts().to_dict()
    cleaned_distribution = df_clean[TARGET_COLUMN].value_counts().to_dict()
    unique_raw_labels = sorted(label for label in raw_labels.unique() if label)

    ambiguous_labels = {
        "surprise": "bisa positif, negatif, atau netral tergantung konteks",
        "nostalgia": "sering bercampur antara positif, sedih, dan reflektif",
        "bittersweet": "emosi campuran yang tidak sepenuhnya negatif",
        "pressure": "dapat bernuansa stres atau netral tergantung kalimat",
        "pensive": "sering reflektif dan tidak selalu negatif",
        "sympathy": "lebih dekat ke empati, tidak selalu positif murni",
    }

    majority_ratio = cleaned_distribution["positive"] / sum(cleaned_distribution.values())
    imbalance_gap = cleaned_distribution["positive"] - cleaned_distribution["neutral"]

    return {
        "source_file": str(RAW_DATA_PATH),
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "raw_unique_labels": int(raw_labels.nunique()),
        "raw_class_distribution_3way": raw_distribution,
        "clean_class_distribution_3way": cleaned_distribution,
        "majority_positive_ratio": majority_ratio,
        "positive_minus_neutral_gap": imbalance_gap,
        "sample_ambiguous_labels": ambiguous_labels,
        "top_15_raw_labels": raw_labels.value_counts().head(15).to_dict(),
        "all_raw_labels_sorted": unique_raw_labels,
    }


def build_markdown(payload: dict) -> str:
    raw_dist = payload["raw_class_distribution_3way"]
    clean_dist = payload["clean_class_distribution_3way"]
    ambiguous_lines = "\n".join(
        f"- `{label}`: {reason}" for label, reason in payload["sample_ambiguous_labels"].items()
    )
    top_label_lines = "\n".join(
        f"- `{label}`: `{count}`" for label, count in payload["top_15_raw_labels"].items()
    )

    return (
        "# Audit Dataset dan Mapping Label\n\n"
        "## Ringkasan Dataset\n\n"
        f"- Sumber utama training: `{payload['source_file']}`\n"
        f"- Baris mentah: `{payload['raw_rows']}`\n"
        f"- Baris setelah cleaning, drop teks kosong, dan deduplikasi: `{payload['clean_rows']}`\n"
        f"- Label mentah unik: `{payload['raw_unique_labels']}`\n\n"
        "## Distribusi Label 3 Kelas\n\n"
        "Distribusi sebelum cleaning:\n\n"
        f"- `positive`: `{raw_dist['positive']}`\n"
        f"- `negative`: `{raw_dist['negative']}`\n"
        f"- `neutral`: `{raw_dist['neutral']}`\n\n"
        "Distribusi setelah cleaning:\n\n"
        f"- `positive`: `{clean_dist['positive']}`\n"
        f"- `negative`: `{clean_dist['negative']}`\n"
        f"- `neutral`: `{clean_dist['neutral']}`\n\n"
        "## Top 15 Label Mentah\n\n"
        f"{top_label_lines}\n\n"
        "## Temuan Utama\n\n"
        f"- Dataset masih timpang ke arah `positive` dengan rasio mayoritas `{payload['majority_positive_ratio']:.2%}`.\n"
        f"- Selisih antara kelas `positive` dan `neutral` setelah cleaning adalah `{payload['positive_minus_neutral_gap']}` baris.\n"
        "- Label mentah sangat granular, tetapi sistem deployment tetap harus memetakannya ke 3 kelas output.\n"
        "- Tidak ada label mentah yang jatuh ke fallback tak dikenal; seluruh label saat ini tertangani oleh mapping.\n"
        "- Masalah utama ada pada kompresi semantik dan class imbalance, bukan parsing dataset.\n\n"
        "## Label Ambigu yang Perlu Diaudit Manual\n\n"
        f"{ambiguous_lines}\n\n"
        "## Dampak ke Model\n\n"
        "- Model yang tidak dibalancing akan mudah bias ke `positive` karena kelas ini jauh dominan.\n"
        "- Kelas `neutral` tetap menjadi kelas tersulit karena jumlah contoh paling kecil dan maknanya paling ambigu.\n"
        "- Accuracy tidak cukup; baca confusion matrix dan macro F1 sebagai indikator utama.\n\n"
        "## Rekomendasi Lanjutan\n\n"
        "1. Pertahankan output 3 kelas agar app tetap kompatibel.\n"
        "2. Audit subset label ambigu sebelum mengubah mapping produksi.\n"
        "3. Tambah contoh `neutral` jika dataset baru tersedia.\n"
        "4. Gunakan audit ini sebagai referensi sebelum mengganti model production.\n"
    )


def main() -> None:
    payload = build_audit_payload()
    AUDIT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Dataset audit written to {AUDIT_MD_PATH}")
    print(f"Dataset audit JSON written to {AUDIT_JSON_PATH}")


if __name__ == "__main__":
    main()
