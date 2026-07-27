# ai_models/downloader.py
from pathlib import Path
from huggingface_hub import snapshot_download

def download_hf_model(
    repo_id: str,
    local_model_name: str | None = None,
) -> Path:

    try:

        ai_models_dir = Path(".")
        ai_models_dir.mkdir(parents=True, exist_ok=True)

        if not local_model_name:
            local_model_name = repo_id.split("/")[-1]

        local_model_path = ai_models_dir / local_model_name

        print(f"Downloading: {repo_id}")
        print(f"Saving to: {local_model_path}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_model_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print("Download completed!")

        return local_model_path

    except Exception as e:
        raise RuntimeError(
            f"Failed downloading model '{repo_id}': {e}"
        ) from e


if __name__ == "__main__":
    download_hf_model(repo_id="Qwen/Qwen2.5-0.5B-Instruct")