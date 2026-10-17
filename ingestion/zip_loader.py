"""
ingestion/zip_loader.py

Handles ZIP archive extraction and ingestion of repository content.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from loguru import logger

from config import IngestionConfig, get_config
from ingestion.file_loader import FileDocument, load_file


class ZipLoader:
    """
    Extracts a ZIP archive to a temporary directory and yields
    FileDocument objects for every supported file.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or get_config().ingestion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_zip(self, zip_path: Path) -> list[FileDocument]:
        """
        Extract and load all supported files from a ZIP archive.

        Args:
            zip_path: Path to the .zip file.

        Returns:
            List of FileDocument objects.

        Raises:
            ValueError: If the file is not a valid ZIP archive.
            RuntimeError: If extraction fails.
        """
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"{zip_path} is not a valid ZIP file.")

        temp_root = self.config.temp_dir.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        extract_dir = Path(tempfile.mkdtemp(dir=temp_root)).resolve()
        try:
            self._extract(zip_path, extract_dir)
            root = self._find_root(extract_dir)
            docs = self._walk_directory(root)
            logger.info(f"Loaded {len(docs)} files from ZIP: {zip_path.name}")
            return docs
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            logger.debug(f"Cleaned up extract dir: {extract_dir}")

    def load_from_bytes(self, data: bytes, filename: str = "upload.zip") -> list[FileDocument]:
        """
        Load repository from raw ZIP bytes (e.g. Streamlit UploadedFile).

        Args:
            data:     Raw bytes of the ZIP file.
            filename: Original filename (for logging).

        Returns:
            List of FileDocument objects.
        """
        temp_root = self.config.temp_dir.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        tmp_zip = Path(tempfile.mktemp(suffix=".zip", dir=temp_root)).resolve()
        try:
            tmp_zip.write_bytes(data)
            return self.load_from_zip(tmp_zip)
        finally:
            tmp_zip.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(zip_path: Path, dest: Path) -> None:
        """Safely extract a ZIP file, guarding against path traversal."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Guard against absolute paths and ".." traversal
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    logger.warning(f"Skipping unsafe ZIP entry: {member.filename}")
                    continue
                zf.extract(member, dest)

    @staticmethod
    def _find_root(extract_dir: Path) -> Path:
        """
        If the ZIP contains a single top-level folder (common pattern),
        return that folder as the effective root.
        """
        children = [p for p in extract_dir.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return extract_dir

    def _walk_directory(self, root: Path) -> list[FileDocument]:
        """Walk *root* and collect FileDocument objects."""
        docs: list[FileDocument] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if self._should_ignore(path, root):
                continue
            if path.suffix.lower() not in self.config.supported_extensions:
                continue

            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.debug(f"Skipping large file {path} ({size_mb:.1f} MB)")
                continue

            doc = load_file(path, root)
            if doc is not None:
                docs.append(doc)
        return docs

    def _should_ignore(self, path: Path, root: Path) -> bool:
        relative = path.relative_to(root)
        return bool(set(relative.parts) & set(self.config.ignored_dirs))
