"""
ingestion/file_loader.py

Modular file loader architecture.
Each concrete loader knows how to read a specific file type and attach
rich metadata (language, encoding, etc.).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chardet
from loguru import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FileDocument:
    """Represents the raw content of a single source file."""

    filename: str
    filepath: str          # relative to repo root
    extension: str
    content: str
    language: str
    size_bytes: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "extension": self.extension,
            "language": self.language,
            "size_bytes": self.size_bytes,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Base loader
# ---------------------------------------------------------------------------

class BaseLoader(ABC):
    """Abstract base class for all file loaders."""

    SUPPORTED_EXTENSIONS: list[str] = []

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    @abstractmethod
    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        """Load a file and return a FileDocument, or None on failure."""
        ...

    def _read_text(self, path: Path) -> Optional[str]:
        """Read file text, auto-detecting encoding."""
        try:
            raw = path.read_bytes()
            detection = chardet.detect(raw)
            encoding = detection.get("encoding") or "utf-8"
            return raw.decode(encoding, errors="replace")
        except Exception as exc:
            logger.warning(f"Could not read {path}: {exc}")
            return None

    def _relative_path(self, path: Path, root: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)


# ---------------------------------------------------------------------------
# Concrete loaders
# ---------------------------------------------------------------------------

class NotebookLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".ipynb"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None

        try:
            notebook = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid notebook JSON in {path}: {exc}")
            return None

        cells: list[str] = []
        for index, cell in enumerate(notebook.get("cells", []), 1):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not isinstance(source, str) or not source.strip():
                continue
            cells.append(f"# Cell {index} ({cell_type})\n{source.rstrip()}")

        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".ipynb",
            content="\n\n".join(cells),
            language="python",
            size_bytes=path.stat().st_size,
            metadata={
                "cell_count": len(notebook.get("cells", [])),
                "content_cell_count": len(cells),
                "notebook_format": notebook.get("nbformat"),
            },
        )


class PythonLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".py", ".pyw"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language="python",
            size_bytes=path.stat().st_size,
            metadata={"has_classes": "class " in content, "has_functions": "def " in content},
        )


class JavaScriptLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".js", ".jsx", ".mjs", ".cjs"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language="javascript",
            size_bytes=path.stat().st_size,
        )


class TypeScriptLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".ts", ".tsx"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language="typescript",
            size_bytes=path.stat().st_size,
        )


class JavaLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".java"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".java",
            content=content,
            language="java",
            size_bytes=path.stat().st_size,
        )


class CppLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".cpp", ".c", ".cc", ".cxx", ".h", ".hpp"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        lang = "c" if path.suffix in {".c", ".h"} else "cpp"
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language=lang,
            size_bytes=path.stat().st_size,
        )


class GoLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".go"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".go",
            content=content,
            language="go",
            size_bytes=path.stat().st_size,
        )


class RustLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".rs"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".rs",
            content=content,
            language="rust",
            size_bytes=path.stat().st_size,
        )


class PhpLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".php"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".php",
            content=content,
            language="php",
            size_bytes=path.stat().st_size,
        )


class MarkdownLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".md", ".mdx"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language="markdown",
            size_bytes=path.stat().st_size,
        )


class JsonLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".json"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        # Validate JSON
        try:
            parsed = json.loads(content)
            metadata = {"valid_json": True, "keys": list(parsed.keys()) if isinstance(parsed, dict) else []}
        except json.JSONDecodeError:
            metadata = {"valid_json": False}

        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".json",
            content=content,
            language="json",
            size_bytes=path.stat().st_size,
            metadata=metadata,
        )


class YamlLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".yaml", ".yml"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language="yaml",
            size_bytes=path.stat().st_size,
        )


class SqlLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".sql"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=".sql",
            content=content,
            language="sql",
            size_bytes=path.stat().st_size,
        )


class GenericTextLoader(BaseLoader):
    """Fallback loader for any text-based file."""

    SUPPORTED_EXTENSIONS = [".txt", ".html", ".css", ".env", ".cfg", ".ini", ".toml"]

    def load(self, path: Path, root: Path) -> Optional[FileDocument]:
        content = self._read_text(path)
        if content is None:
            return None
        ext_map = {
            ".html": "html", ".css": "css", ".toml": "toml",
            ".cfg": "config", ".ini": "config", ".env": "config",
        }
        lang = ext_map.get(path.suffix.lower(), "text")
        return FileDocument(
            filename=path.name,
            filepath=self._relative_path(path, root),
            extension=path.suffix.lower(),
            content=content,
            language=lang,
            size_bytes=path.stat().st_size,
        )


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

_LOADERS: list[BaseLoader] = [
    NotebookLoader(),
    PythonLoader(),
    JavaScriptLoader(),
    TypeScriptLoader(),
    JavaLoader(),
    CppLoader(),
    GoLoader(),
    RustLoader(),
    PhpLoader(),
    MarkdownLoader(),
    JsonLoader(),
    YamlLoader(),
    SqlLoader(),
    GenericTextLoader(),  # must be last – it's the fallback
]


def load_file(path: Path, root: Path) -> Optional[FileDocument]:
    """
    Find the right loader for *path* and return a FileDocument.
    Returns None if no loader matches or the file cannot be read.
    """
    for loader in _LOADERS:
        if loader.can_load(path):
            try:
                return loader.load(path, root)
            except Exception as exc:
                logger.warning(f"Loader {loader.__class__.__name__} failed for {path}: {exc}")
                return None
    logger.debug(f"No loader found for {path}")
    return None
