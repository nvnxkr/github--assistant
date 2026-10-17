"""
tests/test_file_loader.py

Unit tests for the file loader system.
"""

import tempfile
import json
from pathlib import Path

import pytest
from ingestion.file_loader import (
    FileDocument,
    GenericTextLoader,
    JsonLoader,
    MarkdownLoader,
    PythonLoader,
    load_file,
)


def write_tmp(content: str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


class TestPythonLoader:
    def test_loads_python_file(self):
        path = write_tmp("def hello():\n    return 'world'\n", ".py")
        loader = PythonLoader()
        doc = loader.load(path, path.parent)
        assert doc is not None
        assert doc.language == "python"
        assert "def hello" in doc.content
        path.unlink()

    def test_detects_functions(self):
        path = write_tmp("def foo(): pass\n", ".py")
        loader = PythonLoader()
        doc = loader.load(path, path.parent)
        assert doc.metadata.get("has_functions") is True
        path.unlink()


class TestMarkdownLoader:
    def test_loads_md(self):
        path = write_tmp("# Title\nSome content.\n", ".md")
        loader = MarkdownLoader()
        doc = loader.load(path, path.parent)
        assert doc is not None
        assert doc.language == "markdown"
        path.unlink()


class TestJsonLoader:
    def test_valid_json(self):
        path = write_tmp('{"name": "test", "version": "1.0"}', ".json")
        loader = JsonLoader()
        doc = loader.load(path, path.parent)
        assert doc is not None
        assert doc.metadata.get("valid_json") is True
        assert "name" in doc.metadata.get("keys", [])
        path.unlink()

    def test_invalid_json(self):
        path = write_tmp("{not valid json", ".json")
        loader = JsonLoader()
        doc = loader.load(path, path.parent)
        assert doc is not None
        assert doc.metadata.get("valid_json") is False
        path.unlink()


class TestLoadFile:
    def test_dispatches_python(self):
        path = write_tmp("x = 1\n", ".py")
        doc = load_file(path, path.parent)
        assert doc is not None
        assert doc.language == "python"
        path.unlink()

    def test_dispatches_markdown(self):
        path = write_tmp("# Hello\n", ".md")
        doc = load_file(path, path.parent)
        assert doc is not None
        assert doc.language == "markdown"
        path.unlink()

    def test_unknown_extension_returns_none(self):
        path = write_tmp("binary", ".xyz")
        doc = load_file(path, path.parent)
        # .xyz is not in any supported extension list
        assert doc is None
        path.unlink()

    def test_dispatches_notebook(self):
        path = write_tmp(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["# Analysis\n"]},
                        {"cell_type": "code", "source": ["value = 42\n", "print(value)\n"]},
                        {"cell_type": "code", "source": []},
                    ],
                    "metadata": {},
                    "nbformat": 4,
                }
            ),
            ".ipynb",
        )

        doc = load_file(path, path.parent)

        assert doc is not None
        assert doc.language == "python"
        assert "# Cell 1 (markdown)" in doc.content
        assert "value = 42" in doc.content
        assert "print(value)" in doc.content
        assert "output" not in doc.content
        assert doc.metadata["cell_count"] == 3
        assert doc.metadata["content_cell_count"] == 2
        path.unlink()
