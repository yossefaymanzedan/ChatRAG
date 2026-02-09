import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str, cache_dir: str = ".rag/models", offline: bool = False) -> None:
        self.offline = bool(offline)
        if self.offline:
            # Keep old behavior when offline mode is explicitly requested.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            model_ref = self._resolve_model(model_name, cache_dir)
            self.model = SentenceTransformer(model_ref, local_files_only=True)
            return

        # Online mode: allow fetching directly from Hugging Face and cache locally.
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            local_files_only=False,
        )

    def _resolve_model(self, model_name: str, cache_dir: str) -> str:
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            return str(model_path)

        local_model = Path(cache_dir) / model_name.replace("/", "--")
        if local_model.exists() and any(local_model.iterdir()):
            return str(local_model)

        raise RuntimeError(
            f"Offline embedding model not found: {local_model}. "
            "Place the model there before starting the server."
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
