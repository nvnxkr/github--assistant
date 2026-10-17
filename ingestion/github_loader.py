"""
ingestion/github_loader.py

Handles cloning GitHub repositories and extracting file content.
Supports both public repos and private repos (via GitHub token).
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse

import git
from loguru import logger

from config import IngestionConfig, get_config
from ingestion.file_loader import FileDocument, load_file


class GitHubLoader:
    """
    Clones a GitHub repository and yields FileDocument objects
    for every supported source file.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or get_config().ingestion
        self._temp_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_url(self, url: str) -> list[FileDocument]:
        """
        Clone a GitHub URL and return all supported documents.

        Args:
            url: GitHub repository URL, e.g.
                 https://github.com/owner/repo
                 or git@github.com:owner/repo.git

        Returns:
            List of FileDocument objects.
        """
        url = self._normalise_url(url)
        logger.info(f"Cloning repository: {url}")

        clone_dir = self._clone_repository(url)
        try:
            docs = list(self._walk_repository(clone_dir))
            logger.info(f"Loaded {len(docs)} files from {url}")
            return docs
        finally:
            self._cleanup(clone_dir)

    def load_from_local(self, path: Path) -> list[FileDocument]:
        """
        Load an already-cloned / extracted repository from disk.

        Args:
            path: Root directory of the repository.

        Returns:
            List of FileDocument objects.
        """
        logger.info(f"Loading repository from local path: {path}")
        docs = list(self._walk_repository(path))
        logger.info(f"Loaded {len(docs)} files from {path}")
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_url(self, url: str) -> str:
        """Ensure HTTPS URL format and inject token if available."""
        url = url.strip().rstrip("/")
        if not url.endswith(".git"):
            url = url + ".git"

        if self.config.github_token and url.startswith("https://"):
            parsed = urlparse(url)
            url = f"https://{self.config.github_token}@{parsed.netloc}{parsed.path}"

        return url

    def _clone_repository(self, url: str) -> Path:
        """Clone the repository into a temporary directory."""
        temp_root = self.config.temp_dir.resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=temp_root)).resolve()
        try:
            git.Repo.clone_from(
                url,
                temp_dir,
                depth=1,  # shallow clone – faster, less disk
                no_single_branch=True,
            )
            logger.debug(f"Cloned to {temp_dir}")
            return temp_dir
        except git.exc.GitCommandError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone {url}: {exc}") from exc

    def _walk_repository(self, root: Path) -> Generator[FileDocument, None, None]:
        """Recursively walk a repo directory and yield FileDocument objects."""
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if self._should_ignore(file_path, root):
                continue
            if file_path.suffix.lower() not in self.config.supported_extensions:
                continue

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.debug(f"Skipping large file: {file_path} ({file_size_mb:.1f} MB)")
                continue

            doc = load_file(file_path, root)
            if doc is not None:
                yield doc

    def _should_ignore(self, file_path: Path, root: Path) -> bool:
        """Return True if any path component is in the ignore list."""
        relative = file_path.relative_to(root)
        parts = set(relative.parts)
        return bool(parts & set(self.config.ignored_dirs))

    @staticmethod
    def _cleanup(path: Path) -> None:
        """Remove the temporary clone directory."""
        shutil.rmtree(path, ignore_errors=True)
        logger.debug(f"Cleaned up {path}")


def validate_github_url(url: str) -> bool:
    """Return True if *url* looks like a valid GitHub repository URL."""
    pattern = r"^https?://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?$"
    return bool(re.match(pattern, url.strip()))
