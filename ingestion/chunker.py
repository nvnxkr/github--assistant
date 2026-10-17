"""
ingestion/chunker.py

Language-aware code chunking.
Instead of dumb character splits, we split on meaningful code boundaries
(class / function definitions) so each chunk has semantic coherence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from loguru import logger

from config import IngestionConfig, get_config
from ingestion.file_loader import FileDocument


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A semantically meaningful chunk of source code."""

    content: str
    file_name: str
    file_path: str
    language: str
    chunk_type: str          # "function", "class", "module", "block"
    chunk_index: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    extra_metadata: dict = field(default_factory=dict)

    def to_langchain_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "file_name": self.file_name,
                "file_path": self.file_path,
                "language": self.language,
                "chunk_type": self.chunk_type,
                "chunk_index": self.chunk_index,
                "function_name": self.function_name or "",
                "class_name": self.class_name or "",
                "start_line": self.start_line,
                "end_line": self.end_line,
                **self.extra_metadata,
            },
        )


# ---------------------------------------------------------------------------
# Language-specific splitters
# ---------------------------------------------------------------------------

class BaseChunker:
    """Base class for language-specific chunkers."""

    def __init__(self, max_chunk_size: int = 1500, overlap: int = 200) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, doc: FileDocument) -> list[CodeChunk]:
        raise NotImplementedError

    def _split_by_size(
        self,
        content: str,
        file_name: str,
        file_path: str,
        language: str,
        start_index: int = 0,
    ) -> list[CodeChunk]:
        """Fall back to size-based chunking with overlap."""
        chunks = []
        lines = content.splitlines(keepends=True)
        current: list[str] = []
        current_size = 0
        chunk_idx = start_index

        for line in lines:
            current.append(line)
            current_size += len(line)
            if current_size >= self.max_chunk_size:
                chunk_text = "".join(current)
                chunks.append(
                    CodeChunk(
                        content=chunk_text,
                        file_name=file_name,
                        file_path=file_path,
                        language=language,
                        chunk_type="block",
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1
                # Keep overlap lines
                overlap_chars = 0
                overlap_lines: list[str] = []
                for l in reversed(current):
                    overlap_chars += len(l)
                    overlap_lines.insert(0, l)
                    if overlap_chars >= self.overlap:
                        break
                current = overlap_lines
                current_size = sum(len(l) for l in current)

        if current:
            chunks.append(
                CodeChunk(
                    content="".join(current),
                    file_name=file_name,
                    file_path=file_path,
                    language=language,
                    chunk_type="block",
                    chunk_index=chunk_idx,
                )
            )
        return chunks


class PythonChunker(BaseChunker):
    """
    Split Python files on class/function boundaries using regex.
    Preserves the full function/class body in one chunk.
    """

    # Matches top-level and indented class/function definitions
    _SPLIT_RE = re.compile(
        r"^((?:class|def)\s+\w+[^\n]*:)",
        re.MULTILINE,
    )

    def chunk(self, doc: FileDocument) -> list[CodeChunk]:
        content = doc.content
        splits = list(self._SPLIT_RE.finditer(content))

        if not splits:
            return self._split_by_size(content, doc.filename, doc.filepath, "python")

        chunks: list[CodeChunk] = []

        # Content before first split → module-level block
        if splits[0].start() > 0:
            header = content[: splits[0].start()]
            if header.strip():
                chunks.append(
                    CodeChunk(
                        content=header,
                        file_name=doc.filename,
                        file_path=doc.filepath,
                        language="python",
                        chunk_type="module",
                        chunk_index=0,
                    )
                )

        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
            body = content[start:end]

            definition = match.group(1)
            chunk_type = "class" if definition.startswith("class") else "function"
            name_match = re.match(r"(?:class|def)\s+(\w+)", definition)
            name = name_match.group(1) if name_match else None

            start_line = content[:start].count("\n") + 1
            end_line = start_line + body.count("\n")

            # If chunk is too large, sub-split it
            if len(body) > self.max_chunk_size * 1.5:
                sub = self._split_by_size(body, doc.filename, doc.filepath, "python", len(chunks))
                chunks.extend(sub)
            else:
                chunks.append(
                    CodeChunk(
                        content=body,
                        file_name=doc.filename,
                        file_path=doc.filepath,
                        language="python",
                        chunk_type=chunk_type,
                        chunk_index=len(chunks),
                        function_name=name if chunk_type == "function" else None,
                        class_name=name if chunk_type == "class" else None,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        return chunks


class JavaScriptChunker(BaseChunker):
    """
    Split JS/TS files on function / class boundaries.
    Handles function declarations, arrow functions, and class bodies.
    """

    _SPLIT_RE = re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)",
        re.MULTILINE,
    )

    def chunk(self, doc: FileDocument) -> list[CodeChunk]:
        content = doc.content
        splits = list(self._SPLIT_RE.finditer(content))

        if not splits:
            return self._split_by_size(content, doc.filename, doc.filepath, doc.language)

        chunks: list[CodeChunk] = []

        if splits[0].start() > 0:
            header = content[: splits[0].start()]
            if header.strip():
                chunks.append(
                    CodeChunk(
                        content=header,
                        file_name=doc.filename,
                        file_path=doc.filepath,
                        language=doc.language,
                        chunk_type="module",
                        chunk_index=0,
                    )
                )

        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
            body = content[start:end]

            definition = match.group(0)
            chunk_type = "class" if "class " in definition else "function"
            name_match = re.search(r"(?:function|class|const)\s+(\w+)", definition)
            name = name_match.group(1) if name_match else None

            chunks.append(
                CodeChunk(
                    content=body,
                    file_name=doc.filename,
                    file_path=doc.filepath,
                    language=doc.language,
                    chunk_type=chunk_type,
                    chunk_index=len(chunks),
                    function_name=name if chunk_type == "function" else None,
                    class_name=name if chunk_type == "class" else None,
                    start_line=content[:start].count("\n") + 1,
                    end_line=content[:end].count("\n") + 1,
                )
            )

        return chunks


class JavaChunker(BaseChunker):
    """Split Java files on class and method boundaries."""

    _CLASS_RE = re.compile(r"^(?:public\s+|private\s+|protected\s+)?class\s+\w+", re.MULTILINE)
    _METHOD_RE = re.compile(
        r"^\s+(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{",
        re.MULTILINE,
    )

    def chunk(self, doc: FileDocument) -> list[CodeChunk]:
        return self._split_by_size(doc.content, doc.filename, doc.filepath, "java")


class GenericChunker(BaseChunker):
    """Fallback: size-based chunking with overlap."""

    def chunk(self, doc: FileDocument) -> list[CodeChunk]:
        return self._split_by_size(doc.content, doc.filename, doc.filepath, doc.language)


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

_CHUNKER_MAP: dict[str, BaseChunker] = {
    "python": PythonChunker(),
    "javascript": JavaScriptChunker(),
    "typescript": JavaScriptChunker(),
    "java": JavaChunker(),
}
_GENERIC_CHUNKER = GenericChunker()


class CodeChunker:
    """
    Dispatcher that selects the right language-specific chunker
    and returns LangChain Documents.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        cfg = config or get_config().ingestion
        # Update max chunk size from config
        for c in list(_CHUNKER_MAP.values()) + [_GENERIC_CHUNKER]:
            c.max_chunk_size = cfg.chunk_size
            c.overlap = cfg.chunk_overlap

    def chunk_document(self, doc: FileDocument) -> list[Document]:
        """Chunk a FileDocument and return LangChain Documents."""
        chunker = _CHUNKER_MAP.get(doc.language, _GENERIC_CHUNKER)
        try:
            code_chunks = chunker.chunk(doc)
        except Exception as exc:
            logger.warning(f"Chunker failed for {doc.filepath}: {exc}. Falling back.")
            code_chunks = _GENERIC_CHUNKER.chunk(doc)

        logger.debug(f"{doc.filepath}: {len(code_chunks)} chunks ({doc.language})")
        return [c.to_langchain_document() for c in code_chunks]

    def chunk_documents(self, docs: list[FileDocument]) -> list[Document]:
        """Chunk a list of FileDocuments."""
        all_chunks: list[Document] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks created: {len(all_chunks)} from {len(docs)} files")
        return all_chunks
