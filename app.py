"""
app.py — Main orchestrator. 
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from loguru import logger

from chains.rag_chain import RAGChain, build_llm
from config import AppConfig, get_config
from database.logger import DatabaseLogger, Timer
from embeddings.embedding_manager import EmbeddingManager
from ingestion.chunker import CodeChunker
from ingestion.github_loader import GitHubLoader
from ingestion.zip_loader import ZipLoader
from retrieval.retriever import ContextualRetriever
from vectorstore.faiss_manager import FAISSManager
from agents.supervisor import Supervisor


@dataclass
class RepositoryMetadata:
    source: str = ""
    file_count: int = 0
    chunk_count: int = 0
    language_breakdown: dict[str, int] = field(default_factory=dict)
    embedding_model: str = ""
    indexed_at: str = ""
    total_size_mb: float = 0.0


class GitHubRAGAssistant:
    """Top-level orchestrator."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.config.ensure_directories()
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._vector_manager: Optional[FAISSManager] = None
        self._retriever: Optional[ContextualRetriever] = None
        self._rag_chain: Optional[RAGChain] = None
        self._supervisor: Optional[Supervisor] = None
        self._db_logger: Optional[DatabaseLogger] = None
        self.repo_metadata = RepositoryMetadata()
        logger.info(f"GitHubRAGAssistant v{self.config.app_version} ready")

    # ---------- lazy accessors ----------

    @property
    def db_logger(self) -> DatabaseLogger:
        if self._db_logger is None:
            self._db_logger = DatabaseLogger(self.config.database)
        return self._db_logger

    @property
    def embedding_manager(self) -> EmbeddingManager:
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(self.config.embedding)
        return self._embedding_manager

    def _ensure_vector_components(self) -> None:
        if self._vector_manager is None:
            embeddings = self.embedding_manager.get_embeddings(use_cache=True)
            self._vector_manager = FAISSManager(embeddings, self.config.vector_store)
        if self._retriever is None:
            self._retriever = ContextualRetriever(self._vector_manager, self.config.vector_store)

    def _ensure_components(self) -> None:
        self._ensure_vector_components()
        if self._rag_chain is None:
            llm = build_llm(self.config.llm)
            self._rag_chain = RAGChain(self._retriever, llm)
        if self._supervisor is None:
            llm = build_llm(self.config.llm)
            self._supervisor = Supervisor(self._retriever, llm)

    def has_indexed_repository(self) -> bool:
        """Return whether this assistant has a real repository available to query."""
        self._ensure_vector_components()
        return self._vector_manager.has_indexed_documents

    # ---------- ingestion ----------

    def ingest_github_url(self, url: str) -> RepositoryMetadata:
        logger.info(f"Ingesting GitHub URL: {url}")
        start = time.perf_counter()
        loader = GitHubLoader(self.config.ingestion)
        docs = loader.load_from_url(url)
        meta = self._index_documents(docs, source=url)
        duration = time.perf_counter() - start
        self.db_logger.log_ingestion(
            repo_url=url,
            file_count=meta.file_count,
            chunk_count=meta.chunk_count,
            embedding_model=self.config.embedding.sentence_transformer_model,
            duration=duration,
        )
        return meta

    def ingest_zip(
        self,
        zip_path: Path | None = None,
        zip_bytes: bytes | None = None,
    ) -> RepositoryMetadata:
        start = time.perf_counter()
        loader = ZipLoader(self.config.ingestion)
        if zip_bytes is not None:
            docs = loader.load_from_bytes(zip_bytes)
            source = "zip_upload"
        elif zip_path is not None:
            docs = loader.load_from_zip(zip_path)
            source = str(zip_path)
        else:
            raise ValueError("Provide zip_path or zip_bytes.")
        meta = self._index_documents(docs, source=source)
        duration = time.perf_counter() - start
        self.db_logger.log_ingestion(
            repo_url=source,
            file_count=meta.file_count,
            chunk_count=meta.chunk_count,
            embedding_model=self.config.embedding.sentence_transformer_model,
            duration=duration,
        )
        return meta

    def _index_documents(self, docs, source: str) -> RepositoryMetadata:
        from datetime import datetime
        chunker = CodeChunker(self.config.ingestion)
        lc_docs = chunker.chunk_documents(docs)
        lang_breakdown: dict[str, int] = {}
        for d in docs:
            lang_breakdown[d.language] = lang_breakdown.get(d.language, 0) + 1

        # Each ingestion represents a new active repository. Replace the
        # previous index so retrieval cannot return documents from older repos.
        if self._vector_manager is not None:
            self._vector_manager.reset()
        self.embedding_manager.clear_cache()
        self._vector_manager = None
        self._retriever = None
        self._rag_chain = None
        self._supervisor = None
        self._ensure_vector_components()
        added = self._vector_manager.add_documents(lc_docs)
        self.repo_metadata = RepositoryMetadata(
            source=source,
            file_count=len(docs),
            chunk_count=len(lc_docs),
            language_breakdown=lang_breakdown,
            embedding_model=self.config.embedding.sentence_transformer_model,
            indexed_at=datetime.utcnow().isoformat(),
            total_size_mb=sum(d.size_bytes for d in docs) / (1024 * 1024),
        )
        logger.info(f"Indexed {len(docs)} files → {len(lc_docs)} chunks ({added} new)")
        return self.repo_metadata

    # ---------- query ----------

    def ask(
        self,
        question: str,
        session_id: str = "default",
        use_agent: bool = False,
    ) -> dict[str, Any]:
        if not self.has_indexed_repository():
            raise RuntimeError(
                "No repository has been indexed. Index a GitHub URL or ZIP archive before asking questions."
            )
        self._ensure_components()
        with Timer() as t:
            if use_agent:
                result = self._supervisor.run(question)
                agent = result.get("agent", "supervisor")
            else:
                result = self._rag_chain.ask(question, session_id=session_id)
                agent = "rag_chain"
        sources_meta = [doc.metadata for doc in result.get("sources", [])]
        self.db_logger.log_query(
            session_id=session_id,
            query=question,
            answer=result["answer"],
            sources=sources_meta,
            latency=t.elapsed,
            agent_used=agent,
        )
        return {**result, "agent": agent, "latency": round(t.elapsed, 3)}

    def stream(self, question: str, session_id: str = "default") -> Iterator[str]:
        if not self.has_indexed_repository():
            raise RuntimeError(
                "No repository has been indexed. Index a GitHub URL or ZIP archive before asking questions."
            )
        self._ensure_components()
        return self._rag_chain.stream(question, session_id=session_id)

    def clear_history(self, session_id: str = "default") -> None:
        if self._rag_chain:
            self._rag_chain.clear_history(session_id)

    def reset_index(self) -> None:
        if self._vector_manager:
            self._vector_manager.reset()
        self.embedding_manager.clear_cache()
        self._vector_manager = None
        self._retriever = None
        self._rag_chain = None
        self._supervisor = None
        self.repo_metadata = RepositoryMetadata()
        logger.info("Index reset.")

    def get_stats(self) -> dict:
        vs_stats = self._vector_manager.get_stats() if self._vector_manager else {}
        db_stats = self.db_logger.get_stats()
        return {
            "repository": {
                "source": self.repo_metadata.source,
                "files": self.repo_metadata.file_count,
                "chunks": self.repo_metadata.chunk_count,
                "languages": self.repo_metadata.language_breakdown,
                "size_mb": round(self.repo_metadata.total_size_mb, 2),
                "indexed_at": self.repo_metadata.indexed_at,
            },
            "vector_store": vs_stats,
            "queries": db_stats,
            "config": {
                "llm_provider": self.config.llm.llm_provider,
                "embedding_model": self.config.embedding.sentence_transformer_model,
                "vector_store": "FAISS",
            },
        }
