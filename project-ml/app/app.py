from apps.local.app import demo, predict_sentiment

__all__ = ["demo", "predict_sentiment"]


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("gradio belum terpasang. Jalankan: pip install -r project-ml/app/requirements.txt")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
