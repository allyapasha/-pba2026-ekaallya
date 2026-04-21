from __future__ import annotations

try:
    import gradio as gr
except ImportError:
    gr = None

from src.sentiment_project.inference import load_classic_artifacts, predict_with_classic_pipeline


ARTIFACT_LOAD_ERROR = None
CLASSIC_ARTIFACTS = None

try:
    CLASSIC_ARTIFACTS = load_classic_artifacts()
except Exception as exc:
    ARTIFACT_LOAD_ERROR = str(exc)


EXAMPLES = [
    ["Saya sangat senang dengan produk ini."],
    ["Pelayanannya buruk dan mengecewakan."],
    ["Produk ini biasa saja, tidak istimewa."],
    ["Saya suka kualitasnya, pengirimannya juga cepat."],
    ["Saya tidak puas dengan hasil pembelian ini."],
]

DESCRIPTION = (
    "Model produksi aktif memakai Logistic Regression + TF-IDF + fitur numerik sederhana "
    "dengan output 3 kelas: positive, negative, neutral. "
    "Baseline deep learning dipisahkan di folder eksperimen dan tidak dipakai app ini."
)
if ARTIFACT_LOAD_ERROR:
    DESCRIPTION = f"{DESCRIPTION}\n\nStartup warning: {ARTIFACT_LOAD_ERROR}"
if gr is None:
    DESCRIPTION = f"{DESCRIPTION}\n\nUI warning: gradio belum terpasang di environment ini."


def predict_sentiment(text: str) -> dict:
    if ARTIFACT_LOAD_ERROR:
        return {f"model_unavailable: {ARTIFACT_LOAD_ERROR}": 1.0}
    return predict_with_classic_pipeline(text, artifacts=CLASSIC_ARTIFACTS)


def build_demo():
    if gr is None:
        raise RuntimeError("gradio belum terpasang. Install dependency dengan 'pip install -r project-ml/app/requirements.txt'.")
    return gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(
            label="Masukkan teks",
            placeholder="Contoh: Saya sangat puas dengan produk ini.",
            lines=4,
        ),
        outputs=gr.Label(label="Prediksi sentimen", num_top_classes=3),
        title="Sentiment Analysis Indonesian",
        description=DESCRIPTION,
        examples=EXAMPLES,
        cache_examples=False,
        flagging_mode="never",
    )


demo = build_demo() if gr is not None else None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("gradio belum terpasang. Jalankan: pip install -r project-ml/app/requirements.txt")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

