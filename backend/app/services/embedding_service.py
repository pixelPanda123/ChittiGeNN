"""
Embedding service for generating and managing document embeddings
"""

from __future__ import annotations

from typing import List

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if SentenceTransformer is None:
            raise RuntimeError(
                "SentenceTransformer not available (torch backend missing). "
                "Install a CPU-only wheel or skip embedding features."
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_model()
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.tolist()


embedding_service = EmbeddingService()
