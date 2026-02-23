from pathlib import Path

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, cache_dir: str = ".rag/models", offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        local_model = self._resolve_local_model(model_name)
        if local_model is not None:
            self.model = SentenceTransformer(
                str(local_model),
                cache_folder=str(self.cache_dir),
                local_files_only=True,
            )
            return

        if offline:
            expected = self.cache_dir / model_name.replace("/", "--")
            raise RuntimeError(
                f"Offline embedding model not found: {expected}. "
                "Set EMBEDDING_OFFLINE=0 to allow automatic download from Hugging Face."
            )

        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(self.cache_dir),
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}' from Hugging Face. "
                "Check internet access/model name, or place the model under the embedding cache directory."
            ) from exc

    def _resolve_local_model(self, model_name: str) -> Path | None:
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            return model_path

        local_model = self.cache_dir / model_name.replace("/", "--")
        if local_model.exists() and any(local_model.iterdir()):
            return local_model

        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
