"""
vectorstore/faiss_manager.py

FAISS-backed vector store — the sole vector store for this project.
Supports similarity search, MMR, and hybrid (semantic + keyword) retrieval.
Persists the index to disk so embeddings survive restarts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from config import VectorStoreConfig, get_config


class FAISSManager:
    """
    Manages a FAISS vector store with:
    - Persistent disk storage
    - Incremental indexing (skip already-indexed files)
    - Similarity, MMR, and hybrid search
    """

    def __init__(
        self,
        embeddings: Embeddings,
        config: Optional[VectorStoreConfig] = None,
    ) -> None:
        self.embeddings = embeddings
        self.config = config or get_config().vector_store
        self._index_path: Path = self.config.faiss_index_path
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore = None
        self._indexed_paths: set[str] = set()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _store_path(self) -> str:
        return str(self._index_path / "faiss_index")

    def _meta_path(self) -> Path:
        return self._index_path / "indexed_paths.txt"

    def _load_indexed_paths(self) -> set[str]:
        if self._meta_path().exists():
            return set(self._meta_path().read_text().splitlines())
        return set()

    def _save_indexed_paths(self) -> None:
        self._meta_path().write_text("\n".join(sorted(self._indexed_paths)))

    @property
    def has_indexed_documents(self) -> bool:
        """Return whether at least one real repository file has been indexed."""
        self.get_or_create()
        return bool(self._indexed_paths)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def get_or_create(self):
        """Return the FAISS vectorstore, loading from disk if it exists."""
        if self._vectorstore is not None:
            return self._vectorstore

        from langchain_community.vectorstores import FAISS

        # LangChain FAISS.save_local stores index.faiss and index.pkl
        # inside the directory passed to it.
        store_file = Path(self._store_path()) / "index.faiss"
        if store_file.exists():
            logger.info(f"Loading FAISS index from {self._store_path()}")
            try:
                self._vectorstore = FAISS.load_local(
                    self._store_path(),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._indexed_paths = self._load_indexed_paths()
                logger.info(f"Loaded FAISS index ({len(self._indexed_paths)} files indexed)")
                return self._vectorstore
            except Exception as exc:
                logger.warning(f"Could not load existing index: {exc}. Will create fresh.")

        # Bootstrap with a sentinel document so the store exists
        self._vectorstore = FAISS.from_texts(
            ["__sentinel__"],
            self.embeddings,
            metadatas=[{"file_path": "__sentinel__"}],
        )
        return self._vectorstore

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents, skipping already-indexed file paths.
        Returns number of newly added chunks.
        """
        store = self.get_or_create()

        # Filter out already-indexed files
        new_docs = [
            d for d in documents
            if d.metadata.get("file_path") not in self._indexed_paths
        ]

        if not new_docs:
            logger.info("All files already indexed – nothing new to add.")
            return 0

        logger.info(f"Adding {len(new_docs)} new chunks to FAISS …")

        # Batch in groups of 128 to avoid memory spikes
        batch_size = 128
        added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            store.add_documents(batch)
            added += len(batch)

        # Track newly indexed file paths
        for d in new_docs:
            fp = d.metadata.get("file_path")
            if fp:
                self._indexed_paths.add(fp)

        # Persist
        store.save_local(self._store_path())
        self._save_indexed_paths()
        logger.info(f"FAISS index saved — {added} chunks added, total files: {len(self._indexed_paths)}")
        return added

    def reset(self) -> None:
        """Delete all indexed data."""
        import shutil
        shutil.rmtree(self._index_path, ignore_errors=True)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._vectorstore = None
        self._indexed_paths = set()
        logger.info("FAISS index reset.")

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        k = k or self.config.top_k
        store = self.get_or_create()
        if filter:
            return store.similarity_search(query, k=k, filter=filter)
        return store.similarity_search(query, k=k)

    def mmr_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: Optional[float] = None,
    ) -> list[Document]:
        k = k or self.config.top_k
        lambda_mult = lambda_mult if lambda_mult is not None else self.config.mmr_diversity_score
        store = self.get_or_create()
        return store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[Document]:
        """
        Hybrid: MMR for diversity + keyword re-ranking.
        alpha=1 → pure semantic, alpha=0 → pure keyword.
        """
        k = k or self.config.top_k
        alpha = self.config.hybrid_alpha

        semantic_docs = self.mmr_search(query, k=k * 2)

        query_tokens = set(query.lower().split())

        def kw_score(doc: Document) -> float:
            tokens = set(doc.page_content.lower().split())
            overlap = len(query_tokens & tokens)
            return overlap / max(len(query_tokens), 1)

        scored = [
            (doc, alpha + (1 - alpha) * kw_score(doc))
            for doc in semantic_docs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]

    def get_retriever(self, search_type: str = "hybrid"):
        """Return a LangChain-compatible retriever."""
        store = self.get_or_create()
        if search_type == "similarity":
            return store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.top_k},
            )
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.top_k,
                "fetch_k": self.config.top_k * 4,
                "lambda_mult": self.config.hybrid_alpha,
            },
        )

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        try:
            store = self.get_or_create()
            total_chunks = store.index.ntotal  # type: ignore[attr-defined]
            if not self._indexed_paths and total_chunks == 1:
                total_chunks = 0
        except Exception:
            total_chunks = -1
        return {
            "total_chunks": total_chunks,
            "indexed_files": len(self._indexed_paths),
            "index_path": str(self._index_path),
            "backend": "FAISS",
        }
