"""
retrieval/retriever.py

Retrieval layer wrapping FAISSManager with hybrid search,
deduplication, and citation formatting.
"""

from __future__ import annotations
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from loguru import logger
from pydantic import Field

from config import VectorStoreConfig, get_config
from vectorstore.faiss_manager import FAISSManager


class HybridRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by FAISSManager."""

    vector_manager: Any = Field(description="FAISSManager instance")
    search_type: str = Field(default="hybrid")
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self.search_type == "hybrid":
            return self.vector_manager.hybrid_search(query, k=self.k)
        elif self.search_type == "mmr":
            return self.vector_manager.mmr_search(query, k=self.k)
        return self.vector_manager.similarity_search(query, k=self.k)


class ContextualRetriever:
    """Higher-level retriever with deduplication and citation formatting."""

    def __init__(
        self,
        vector_manager: FAISSManager,
        config: VectorStoreConfig | None = None,
    ) -> None:
        self.vector_manager = vector_manager
        self.config = config or get_config().vector_store

    def retrieve(
        self,
        query: str,
        search_type: str = "hybrid",
        k: int | None = None,
    ) -> list[Document]:
        k = k or self.config.top_k
        if search_type == "hybrid":
            docs = self.vector_manager.hybrid_search(query, k=k)
        elif search_type == "mmr":
            docs = self.vector_manager.mmr_search(query, k=k)
        else:
            docs = self.vector_manager.similarity_search(query, k=k)
        result = self._deduplicate(docs)
        logger.debug(f"Retrieved {len(result)} docs for: {query[:60]}")
        return result

    def get_langchain_retriever(self, search_type: str = "hybrid") -> BaseRetriever:
        return HybridRetriever(
            vector_manager=self.vector_manager,
            search_type=search_type,
            k=self.config.top_k,
        )

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            key = f"{doc.metadata.get('file_path')}:{doc.metadata.get('chunk_index')}"
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    @staticmethod
    def format_citations(docs: list[Document]) -> str:
        lines = []
        for i, doc in enumerate(docs, 1):
            fp = doc.metadata.get("file_path", "unknown")
            start = doc.metadata.get("start_line", "?")
            end = doc.metadata.get("end_line", "?")
            fn = doc.metadata.get("function_name", "")
            lines.append(f"[{i}] {fp}" + (f"::{fn}" if fn else "") + f" (lines {start}–{end})")
        return "\n".join(lines)
