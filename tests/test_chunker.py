"""
tests/test_chunker.py

Unit tests for the language-aware code chunker.
"""

import pytest
from ingestion.chunker import CodeChunker, PythonChunker, JavaScriptChunker
from ingestion.file_loader import FileDocument


def make_doc(content: str, language: str, ext: str = ".py") -> FileDocument:
    return FileDocument(
        filename=f"test{ext}",
        filepath=f"src/test{ext}",
        extension=ext,
        content=content,
        language=language,
        size_bytes=len(content.encode()),
    )


class TestPythonChunker:
    def test_splits_on_function(self):
        code = """import os

def greet(name):
    return f"Hello, {name}"

def farewell(name):
    return f"Goodbye, {name}"
"""
        doc = make_doc(code, "python")
        chunker = PythonChunker(max_chunk_size=500, overlap=50)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        types = {c.chunk_type for c in chunks}
        assert "function" in types

    def test_splits_on_class(self):
        code = """class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"
"""
        doc = make_doc(code, "python")
        chunker = PythonChunker()
        chunks = chunker.chunk(doc)
        assert any(c.chunk_type == "class" for c in chunks)

    def test_empty_file(self):
        doc = make_doc("", "python")
        chunker = PythonChunker()
        chunks = chunker.chunk(doc)
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_module_header(self):
        code = """#!/usr/bin/env python
\"\"\"Module docstring.\"\"\"
import sys
import os

def main():
    pass
"""
        doc = make_doc(code, "python")
        chunker = PythonChunker()
        chunks = chunker.chunk(doc)
        types = {c.chunk_type for c in chunks}
        assert "module" in types or "function" in types


class TestJavaScriptChunker:
    def test_splits_on_function_declaration(self):
        code = """const express = require('express');

function login(req, res) {
    const { username, password } = req.body;
    res.json({ token: 'abc' });
}

function logout(req, res) {
    res.json({ success: true });
}
"""
        doc = make_doc(code, "javascript", ".js")
        chunker = JavaScriptChunker()
        chunks = chunker.chunk(doc)
        assert any(c.chunk_type == "function" for c in chunks)

    def test_splits_on_class(self):
        code = """class UserService {
    constructor(db) { this.db = db; }
    findById(id) { return this.db.find(id); }
}
"""
        doc = make_doc(code, "javascript", ".js")
        chunker = JavaScriptChunker()
        chunks = chunker.chunk(doc)
        assert any(c.chunk_type == "class" for c in chunks)


class TestCodeChunker:
    def test_dispatch_python(self):
        code = "def hello():\n    return 'world'\n"
        doc = make_doc(code, "python")
        chunker = CodeChunker()
        lc_docs = chunker.chunk_document(doc)
        assert len(lc_docs) >= 1
        assert lc_docs[0].metadata["language"] == "python"

    def test_metadata_present(self):
        code = "def add(a, b):\n    return a + b\n"
        doc = make_doc(code, "python")
        chunker = CodeChunker()
        lc_docs = chunker.chunk_document(doc)
        meta = lc_docs[0].metadata
        assert "file_path" in meta
        assert "chunk_type" in meta
        assert "language" in meta

    def test_multiple_documents(self):
        docs = [
            make_doc("def a(): pass\n", "python"),
            make_doc("def b(): pass\n", "python"),
        ]
        chunker = CodeChunker()
        all_chunks = chunker.chunk_documents(docs)
        assert len(all_chunks) >= 2
