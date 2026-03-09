"""
Document Indexer - reads markdown files from docs/agentKT/,
chunks them, computes embeddings, and stores them in ChromaDB.

Provides a `search(query, top_k)` function used by agents to do RAG.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from brain_ai.config import get_config

log = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from previous
            if overlap > 0 and current:
                tail = current[-overlap:]
                current = f"{tail}\n\n{para}"
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def _file_hash(path: Path) -> str:
    """Quick MD5 of file contents for dedup."""
    return hashlib.md5(path.read_bytes()).hexdigest()


class DocIndexer:
    """Manages the ChromaDB collection for agent KT docs."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        vs = cfg["vectorstore"]

        self.persist_dir = Path(vs["persist_directory"]).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = vs["collection_name"]
        self.chunk_size = vs.get("chunk_size", 1000)
        self.chunk_overlap = vs.get("chunk_overlap", 200)
        self.docs_dir = Path(cfg["paths"]["docs_dir"]).resolve()

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Use ChromaDB's built-in default embedding function
        # (it uses all-MiniLM-L6-v2 by default via sentence-transformers)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "DocIndexer ready - collection '%s' has %d docs.",
            self.collection_name,
            self.collection.count(),
        )

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index all markdown files in docs/agentKT/.
        If force=False, skip files whose hash hasn't changed.
        """
        if not self.docs_dir.exists():
            log.warning("Docs directory %s does not exist.", self.docs_dir)
            return {"indexed": 0, "skipped": 0}

        md_files = list(self.docs_dir.rglob("*.md"))
        log.info("Found %d markdown files to index.", len(md_files))

        indexed = 0
        skipped = 0

        for md_file in md_files:
            rel_path = str(md_file.relative_to(self.docs_dir))
            file_hash = _file_hash(md_file)

            # Check if already indexed with same hash
            if not force:
                existing = self.collection.get(
                    where={"source": rel_path},
                    include=["metadatas"],
                )
                if existing and existing["metadatas"]:
                    existing_hash = existing["metadatas"][0].get("file_hash", "")
                    if existing_hash == file_hash:
                        skipped += 1
                        continue

                # Remove old chunks for this file before re-indexing
                if existing and existing["ids"]:
                    self.collection.delete(ids=existing["ids"])

            text = md_file.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)

            if not chunks:
                continue

            ids = [f"{rel_path}::chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": rel_path,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]

            self.collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            indexed += 1

        summary = {
            "total_files": len(md_files),
            "indexed": indexed,
            "skipped": skipped,
            "total_chunks": self.collection.count(),
        }
        log.info("Indexing complete: %s", summary)
        return summary

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed docs.
        Returns list of {text, source, score} dicts.
        """
        if self.collection.count() == 0:
            log.warning("Collection is empty. Run indexing first.")
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": 1 - dist,  # cosine distance -> similarity
            })

        return hits
