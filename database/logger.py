"""
database/logger.py

SQLite-based query logger.
Stores every query, retrieved chunks, response, and latency
for audit trails and RAGAS evaluation.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from loguru import logger as log

from config import DatabaseConfig, get_config


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """Stores each user query and the system's response."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text)
    agent_used = Column(String(64))
    retrieved_chunk_count = Column(Integer, default=0)
    retrieved_sources = Column(Text)           # JSON list of file paths
    latency_seconds = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, default="{}")


class IngestionLog(Base):
    """Records repository ingestion events."""

    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_url = Column(String(512))
    file_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    embedding_model = Column(String(128))
    duration_seconds = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="success")
    error_message = Column(Text)


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class DatabaseLogger:
    """Thread-safe SQLite logger using SQLAlchemy."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        cfg = config or get_config().database
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        engine_url = f"sqlite:///{cfg.db_path}"
        self._engine = create_engine(engine_url, echo=False)
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)
        log.info(f"SQLite log database at {cfg.db_path}")

    @contextmanager
    def _session(self):
        session: Session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_query(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: list[dict],
        latency: float,
        agent_used: str = "rag_chain",
        extra: dict | None = None,
    ) -> None:
        """Log a query/response pair."""
        source_paths = [s.get("file_path", "") for s in sources]
        with self._session() as db:
            db.add(QueryLog(
                session_id=session_id,
                query=query,
                answer=answer,
                agent_used=agent_used,
                retrieved_chunk_count=len(sources),
                retrieved_sources=json.dumps(source_paths),
                latency_seconds=latency,
                metadata_json=json.dumps(extra or {}),
            ))

    def log_ingestion(
        self,
        repo_url: str,
        file_count: int,
        chunk_count: int,
        embedding_model: str,
        duration: float,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        """Log a repository ingestion event."""
        with self._session() as db:
            db.add(IngestionLog(
                repo_url=repo_url,
                file_count=file_count,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                duration_seconds=duration,
                status=status,
                error_message=error,
            ))

    # ------------------------------------------------------------------
    # Query history
    # ------------------------------------------------------------------

    def get_recent_queries(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._session() as db:
            rows = (
                db.query(QueryLog)
                .filter(QueryLog.session_id == session_id)
                .order_by(QueryLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "query": r.query,
                    "answer": r.answer,
                    "agent": r.agent_used,
                    "latency": r.latency_seconds,
                    "timestamp": str(r.timestamp),
                    "sources": json.loads(r.retrieved_sources or "[]"),
                }
                for r in rows
            ]

    def get_ingestion_history(self, limit: int = 20) -> list[dict]:
        with self._session() as db:
            rows = (
                db.query(IngestionLog)
                .order_by(IngestionLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "repo_url": r.repo_url,
                    "file_count": r.file_count,
                    "chunk_count": r.chunk_count,
                    "duration": r.duration_seconds,
                    "status": r.status,
                    "timestamp": str(r.timestamp),
                }
                for r in rows
            ]

    def get_stats(self) -> dict:
        """Return high-level statistics."""
        with self._session() as db:
            total_queries = db.query(QueryLog).count()
            avg_latency = db.execute(
                text("SELECT AVG(latency_seconds) FROM query_logs")
            ).scalar() or 0
            total_ingestions = db.query(IngestionLog).count()
        return {
            "total_queries": total_queries,
            "avg_latency_seconds": round(avg_latency, 3),
            "total_ingestions": total_ingestions,
        }


# ---------------------------------------------------------------------------
# Convenience timer context manager
# ---------------------------------------------------------------------------

@dataclass
class Timer:
    """Simple wall-clock timer."""
    _start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
