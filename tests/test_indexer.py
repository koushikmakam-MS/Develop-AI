"""Tests for brain_ai.vectorstore.indexer — chunking and search."""

import hashlib

from brain_ai.vectorstore.indexer import _chunk_text, _file_hash

# ── _chunk_text ──────────────────────────────────────────────────────────

class TestChunkText:
    def test_single_paragraph(self):
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_splits_on_paragraph_boundaries(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _chunk_text(text, chunk_size=25, overlap=0)
        assert len(chunks) >= 2
        assert "Para one" in chunks[0]

    def test_overlap_carries_context(self):
        paragraphs = [f"Paragraph {i} with some content here." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks_no_overlap = _chunk_text(text, chunk_size=80, overlap=0)
        chunks_with_overlap = _chunk_text(text, chunk_size=80, overlap=20)
        # With overlap, chunks may share text → possibly more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_empty_text(self):
        chunks = _chunk_text("", chunk_size=100, overlap=0)
        # Should return either empty list or a list with an empty string
        assert all(c == "" for c in chunks) or chunks == []

    def test_very_large_paragraph_not_lost(self):
        """A single paragraph larger than chunk_size should still appear."""
        text = "A" * 500
        chunks = _chunk_text(text, chunk_size=100, overlap=0)
        total = "".join(chunks)
        # All content should be present
        assert len(total) >= 500

    def test_multiple_newlines_as_boundaries(self):
        text = "First.\n\n\n\nSecond.\n\nThird."
        chunks = _chunk_text(text, chunk_size=200, overlap=0)
        full = " ".join(chunks)
        assert "First" in full
        assert "Second" in full
        assert "Third" in full


# ── _file_hash ───────────────────────────────────────────────────────────

class TestFileHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("Hello world")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.md"
        f1.write_text("Content A")
        f2 = tmp_path / "b.md"
        f2.write_text("Content B")
        assert _file_hash(f1) != _file_hash(f2)

    def test_returns_md5_hex(self, tmp_path):
        f = tmp_path / "test.md"
        content = "test content"
        f.write_bytes(content.encode())
        expected = hashlib.md5(content.encode()).hexdigest()
        assert _file_hash(f) == expected
