"""
embeddings/embedding_manager.py

Manages embedding model selection and caching.
Supports Sentence Transformers and Mistral embeddings.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

from langchain_core.embeddings import Embeddings
from loguru import logger

from config import EmbeddingConfig, PROJECT_ROOT, get_config


class EmbeddingCache:
    """
    Simple disk-based cache to avoid re-embedding identical text chunks.
    Stored as a pickled dict: sha256(text) → embedding_vector.
    """

    def __init__(self, cache_path: Path = PROJECT_ROOT / "data" / "embedding_cache.pkl") -> None:
        self.cache_path = cache_path
        self._store: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._store = pickle.load(f)
                logger.debug(f"Loaded {len(self._store)} cached embeddings")
            except Exception as exc:
                logger.warning(f"Could not load embedding cache: {exc}")
                self._store = {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._store, f)
        logger.debug(f"Saved {len(self._store)} embeddings to cache")

    def clear(self) -> None:
        """Delete all cached vectors from memory and disk."""
        self._store.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.debug("Cleared embedding cache")

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        return self._store.get(self._key(text))

    def set(self, text: str, embedding: list[float]) -> None:
        self._store[self._key(text)] = embedding

    def __len__(self) -> int:
        return len(self._store)


class CachedEmbeddings(Embeddings):
    """
    Wrapper that adds caching around any LangChain Embeddings object.
    Cache hits avoid redundant API/model calls.
    """

    def __init__(self, base: Embeddings, cache: EmbeddingCache) -> None:
        self._base = base
        self._cache = cache

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} uncached texts")
            new_embeddings = self._base.embed_documents(uncached_texts)
            for idx, emb, text in zip(uncached_indices, new_embeddings, uncached_texts):
                results[idx] = emb
                self._cache.set(text, emb)
            self._cache.save()

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        emb = self._base.embed_query(text)
        self._cache.set(text, emb)
        self._cache.save()
        return emb


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain adapter for a CPU-loaded Sentence Transformer model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": False},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


class EmbeddingManager:
    """
    Factory that creates the right embedding model based on config.
    Optionally wraps in a CachedEmbeddings layer.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or get_config().embedding
        self._model: Optional[Embeddings] = None

    def get_embeddings(self, use_cache: bool = True) -> Embeddings:
        """
        Return the configured embeddings model, optionally with caching.

        Args:
            use_cache: Whether to wrap in a disk cache.

        Returns:
            LangChain-compatible Embeddings object.
        """
        if self._model is None:
            self._model = self._build_model()

        if use_cache:
            cache = EmbeddingCache()
            return CachedEmbeddings(self._model, cache)
        return self._model

    def clear_cache(self) -> None:
        """Remove cached vectors before indexing a different repository."""
        EmbeddingCache().clear()

    def _build_model(self) -> Embeddings:
        provider = self.config.embedding_provider
        logger.info(f"Initialising embeddings: provider={provider}")

        if provider == "sentence_transformers":
            return self._build_sentence_transformers()
        elif provider == "mistral":
            return self._build_mistral()
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _build_sentence_transformers(self) -> Embeddings:
        try:
            return SentenceTransformerEmbeddings(self.config.sentence_transformer_model)
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use "
                "sentence_transformers embeddings."
            ) from exc

    def _build_mistral(self) -> Embeddings:
        try:
            from langchain_mistralai import MistralAIEmbeddings

            cfg = get_config()
            return MistralAIEmbeddings(
                model=self.config.mistral_embedding_model,
                mistral_api_key=cfg.llm.mistral_api_key,
            )
        except ImportError as exc:
            raise ImportError(
                "Install langchain-mistralai to use Mistral embeddings."
            ) from exc
