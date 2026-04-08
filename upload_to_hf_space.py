import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a Hugging Face Space from a local app folder."
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Hugging Face username or organization name.",
    )
    parser.add_argument(
        "--space-name",
        required=True,
        help="Target Space name, for example sentiment-analysis-indonesian.",
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Local folder containing app.py, README.md, requirements.txt, and model files.",
    )
    parser.add_argument(
        "--sdk",
        default="gradio",
        help="Space SDK type. Default: gradio.",
    )
    parser.add_argument(
        "--space-storage",
        default="small",
        choices=["small", "medium", "large"],
        help="Persistent storage tier if needed. Default: small.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the Space reaches RUNNING or an error state.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Maximum wait time in seconds when --wait is used. Default: 900.",
    )
    return parser.parse_args()


def ensure_token(token: str | None) -> str:
    if token:
        return token
    raise SystemExit("HF token not provided. Use --token or set HF_TOKEN.")


def ensure_folder(folder: str) -> Path:
    path = Path(folder).resolve()
    required_files = [
        "app.py",
        "README.md",
        "requirements.txt",
        "sentiment_model.pkl",
        "tfidf_vectorizer.pkl",
        "label_encoder.pkl",
        "scaler.pkl",
    ]

    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Folder not found: {path}")

    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required files in {path}: {joined}")

    return path


def create_or_reuse_space(
    api: HfApi,
    repo_id: str,
    sdk: str,
    private: bool,
    space_storage: str,
) -> None:
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=sdk,
            private=private,
            exist_ok=True,
            space_storage=space_storage,
        )
    except TypeError:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=sdk,
            private=private,
            exist_ok=True,
        )


def wait_until_running(api: HfApi, repo_id: str, timeout: int) -> str:
    deadline = time.time() + timeout
    last_stage = None

    while time.time() < deadline:
        info = api.space_info(repo_id=repo_id)
        stage = getattr(info.runtime, "stage", None)
        if stage != last_stage:
            print(f"Space stage: {stage}")
            last_stage = stage

        if stage == "RUNNING":
            return stage
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "PAUSED"}:
            return stage

        time.sleep(10)

    return last_stage or "UNKNOWN"


def main() -> int:
    args = parse_args()
    token = ensure_token(args.token)
    folder = ensure_folder(args.folder)
    repo_id = f"{args.username}/{args.space_name}"
    api = HfApi(token=token)

    try:
        user = api.whoami()
    except HfHubHTTPError as exc:
        print(f"Failed to authenticate to Hugging Face: {exc}", file=sys.stderr)
        return 1

    print(f"Authenticated as: {user.get('name')}")
    print(f"Preparing Space: {repo_id}")
    create_or_reuse_space(
        api=api,
        repo_id=repo_id,
        sdk=args.sdk,
        private=args.private,
        space_storage=args.space_storage,
    )

    print(f"Uploading folder: {folder}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Upload Space app from local folder",
    )

    info = api.space_info(repo_id=repo_id)
    host = getattr(info, "host", None)
    print(f"Space repo: https://huggingface.co/spaces/{repo_id}")
    if host:
        print(f"Space host: {host}")

    if args.wait:
        final_stage = wait_until_running(api, repo_id, args.timeout)
        print(f"Final stage: {final_stage}")
        return 0 if final_stage == "RUNNING" else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
