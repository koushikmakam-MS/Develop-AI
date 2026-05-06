"""
Code Indexer - indexes source code files from the BMS repo into ChromaDB.

Reads .cs, .py, .json, .config files from the cloned repo,
extracts meaningful chunks (classes, methods, blocks), and stores them
in a separate ChromaDB collection for the Coder Agent to search.

Designed to run daily or on-demand via `run_code_index.py`.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb

from brain_ai.config import get_config

log = logging.getLogger(__name__)

# Default code file extensions to index
DEFAULT_CODE_EXTENSIONS = [".cs", ".py", ".json", ".config", ".xml", ".yaml", ".yml"]


def _file_hash(path: Path) -> str:
    """Quick MD5 of file contents for dedup."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _extract_csharp_symbols(text: str) -> List[str]:
    """Extract class and method names from C# code for metadata."""
    symbols = []
    # Class/interface/enum declarations
    for m in re.finditer(r"(?:public|private|internal|protected)?\s*(?:static\s+)?(?:abstract\s+)?(?:partial\s+)?(?:class|interface|enum|struct|record)\s+(\w+)", text):
        symbols.append(m.group(1))
    # Method declarations
    for m in re.finditer(r"(?:public|private|internal|protected)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?(?:virtual\s+)?\w[\w<>\[\],\s]*?\s+(\w+)\s*\(", text):
        symbols.append(m.group(1))
    return symbols


# TODO: Add multi-language support — extract identifiers for Python (import/from),
#       Java (package/import), Go (import), etc. Currently C#-only.
def _extract_csharp_namespaces(text: str) -> List[str]:
    """Extract namespace declarations and service identifiers from C# code.

    Returns unique identifiers like:
    ``["Microsoft.Azure.Management.Dpp", "/api/dpp/", "DppInternalProxy"]``

    These are used to build the boundary map: identifier → owning hive.
    """
    identifiers = set()
    # namespace declarations: namespace Microsoft.Azure.Dpp { ... }
    # or file-scoped: namespace Microsoft.Azure.Dpp;
    for m in re.finditer(r'^\s*namespace\s+([\w.]+)', text, re.MULTILINE):
        identifiers.add(m.group(1))
    # API route prefixes:  [Route("api/dpp/...")] or [HttpGet("api/...")]
    for m in re.finditer(r'\[(?:Route|HttpGet|HttpPost|HttpPut|HttpDelete)\s*\(\s*"(/?\w+/\w+)', text):
        route = m.group(1)
        if not route.startswith("/"):
            route = "/" + route
        identifiers.add(route)
    # Proxy/Client/Gateway class declarations that other hives may reference
    for m in re.finditer(r'class\s+([\w]+(?:Proxy|Client|Gateway|Dispatcher))\b', text):
        identifiers.add(m.group(1))
    return sorted(identifiers)


def _extract_python_symbols(text: str) -> List[str]:
    """Extract class and function names from Python code for metadata."""
    symbols = []
    for m in re.finditer(r"^(?:class|def)\s+(\w+)", text, re.MULTILINE):
        symbols.append(m.group(1))
    return symbols


def _chunk_code(text: str, file_ext: str, chunk_size: int = 1500, overlap: int = 300) -> List[Tuple[str, str]]:
    """
    Split source code into meaningful chunks.
    Returns list of (chunk_text, chunk_label) tuples.

    For C# files: tries to split on class/method boundaries.
    For others: splits on blank-line boundaries.
    """
    if not text.strip():
        return []

    chunks: List[Tuple[str, str]] = []

    if file_ext == ".cs":
        # Try to split on class/method boundaries
        # Pattern: line starting with access modifier or [Attribute]
        boundary_pattern = re.compile(
            r"^(?:\s*(?:\[[\w\.\(\)\"=,\s]+\]\s*)*"
            r"(?:public|private|internal|protected|static|abstract|virtual|override|async|partial|sealed)\s+)",
            re.MULTILINE,
        )
        boundaries = [m.start() for m in boundary_pattern.finditer(text)]

        if boundaries:
            # Add start and end
            if boundaries[0] != 0:
                boundaries.insert(0, 0)
            boundaries.append(len(text))

            for i in range(len(boundaries) - 1):
                segment = text[boundaries[i]:boundaries[i + 1]].strip()
                if not segment:
                    continue
                # If segment is too large, further split it
                if len(segment) > chunk_size:
                    sub_chunks = _split_by_size(segment, chunk_size, overlap)
                    for sc in sub_chunks:
                        label = _first_symbol_in(sc, ".cs")
                        chunks.append((sc, label))
                else:
                    label = _first_symbol_in(segment, ".cs")
                    chunks.append((segment, label))
        else:
            # Fallback to size-based splitting
            sub_chunks = _split_by_size(text, chunk_size, overlap)
            for sc in sub_chunks:
                chunks.append((sc, ""))
    else:
        # Generic splitting by blank lines / size
        sub_chunks = _split_by_size(text, chunk_size, overlap)
        for sc in sub_chunks:
            label = _first_symbol_in(sc, file_ext) if file_ext == ".py" else ""
            chunks.append((sc, label))

    return chunks


def _first_symbol_in(text: str, ext: str) -> str:
    """Extract the first class/method name from a chunk for labeling."""
    if ext == ".cs":
        syms = _extract_csharp_symbols(text[:500])
    elif ext == ".py":
        syms = _extract_python_symbols(text[:500])
    else:
        return ""
    return syms[0] if syms else ""


def _split_by_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by line boundaries."""
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) > chunk_size and current:
            chunks.append(current)
            # Keep overlap from end of previous chunk
            tail_lines = current.splitlines(keepends=True)
            overlap_text = ""
            for tl in reversed(tail_lines):
                if len(overlap_text) + len(tl) > overlap:
                    break
                overlap_text = tl + overlap_text
            current = overlap_text + line
        else:
            current += line
    if current.strip():
        chunks.append(current)
    return chunks


class CodeIndexer:
    """Manages the ChromaDB collection for BMS source code."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()

        vs = cfg["vectorstore"]
        self.persist_dir = Path(vs["persist_directory"]).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Separate collection for code (not docs)
        code_cfg = cfg.get("code_index", {})
        self.collection_name = code_cfg.get("collection_name", "bms_code")
        self.chunk_size = code_cfg.get("chunk_size", 1500)
        self.chunk_overlap = code_cfg.get("chunk_overlap", 300)
        self.extensions = code_cfg.get(
            "file_extensions", DEFAULT_CODE_EXTENSIONS
        )
        self.sync_paths = code_cfg.get("sync_paths", [])
        self.repo_clone_dir = Path(cfg["paths"]["repo_clone_dir"]).resolve()

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "CodeIndexer ready - collection '%s' has %d chunks.",
            self.collection_name,
            self.collection.count(),
        )

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Index all source code files from the cloned repo.
        If force=False, skip files whose hash hasn't changed.
        """
        if not self.repo_clone_dir.exists():
            log.warning("Repo clone dir %s does not exist. Run sync first.", self.repo_clone_dir)
            return {"indexed": 0, "skipped": 0, "error": "repo not cloned"}

        # Gather code files from configured sync paths
        code_files: List[Path] = []
        search_roots = []

        log.info("repo_clone_dir = %s  (exists=%s)", self.repo_clone_dir, self.repo_clone_dir.exists())
        log.info("sync_paths config = %s", self.sync_paths)
        log.info("extensions config = %s", self.extensions)

        if self.sync_paths:
            for sp in self.sync_paths:
                src = self.repo_clone_dir / sp
                log.info("Checking sync_path '%s' -> %s  (exists=%s)", sp, src, src.exists())
                if src.exists():
                    search_roots.append(src)
                else:
                    log.warning("Code sync path %s does not exist, skipping.", src)
        else:
            search_roots.append(self.repo_clone_dir)

        log.info("search_roots = %s", search_roots)

        for root in search_roots:
            for ext in self.extensions:
                found = list(root.rglob(f"*{ext}"))
                log.info("  rglob('*%s') under %s -> %d files", ext, root, len(found))
                code_files.extend(found)

        log.info("Total code files before filtering: %d", len(code_files))

        # Filter out common non-essential paths
        skip_dirs = {".git", "node_modules", "bin", "obj", "packages", ".vs",
                     "__pycache__", ".chromadb", "TestResults"}
        code_files = [
            f for f in code_files
            if not any(sd in f.parts for sd in skip_dirs)
        ]

        log.info("Found %d code files to index (after filtering).", len(code_files))

        indexed = 0
        skipped = 0
        total_chunks = 0
        total_files = len(code_files)

        all_namespaces: set = set()

        for file_idx, code_file in enumerate(code_files):
            if (file_idx + 1) % 100 == 0 or file_idx == 0:
                log.info("Progress: %d/%d files  (%d indexed, %d skipped, %d chunks so far)",
                         file_idx + 1, total_files, indexed, skipped, total_chunks)
            try:
                rel_path = str(code_file.relative_to(self.repo_clone_dir))
                file_hash = _file_hash(code_file)

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

                    # Remove old chunks before re-indexing
                    if existing and existing["ids"]:
                        self.collection.delete(ids=existing["ids"])

                text = code_file.read_text(encoding="utf-8", errors="replace")
                ext = code_file.suffix.lower()

                # Extract symbols for metadata
                file_imports = ""
                if ext == ".cs":
                    symbols = _extract_csharp_symbols(text)
                    # Extract namespaces declared in this file
                    file_namespaces = _extract_csharp_namespaces(text)
                    all_namespaces.update(file_namespaces)
                    # Capture using statements so boundary detection works
                    # even when chunks don't contain the using block
                    usings = re.findall(r'^\s*using\s+([\w.]+)\s*;', text, re.MULTILINE)
                    file_imports = ", ".join(usings[:30])  # Cap at 30 to fit metadata
                elif ext == ".py":
                    symbols = _extract_python_symbols(text)
                else:
                    symbols = []

                # Chunk the code
                chunks = _chunk_code(text, ext, self.chunk_size, self.chunk_overlap)
                if not chunks:
                    continue

                ids = [f"{rel_path}::chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "source": rel_path,
                        "file_hash": file_hash,
                        "file_ext": ext,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_label": label,
                        "symbols": ", ".join(symbols[:20]),  # Top symbols for the file
                        "imports": file_imports,  # using/import statements for boundary detection
                    }
                    for i, (_, label) in enumerate(chunks)
                ]
                documents = [chunk_text for chunk_text, _ in chunks]

                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                indexed += 1
                total_chunks += len(chunks)

            except Exception as e:
                log.warning("Failed to index %s: %s", code_file, e)

        summary = {
            "total_files": len(code_files),
            "indexed": indexed,
            "skipped": skipped,
            "total_chunks_added": total_chunks,
            "collection_total": self.collection.count(),
            "namespaces": sorted(all_namespaces),
        }
        log.info("Code indexing complete: %s", summary)
        return summary

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed code.
        Returns list of {text, source, score, symbols, chunk_label} dicts.
        """
        if self.collection.count() == 0:
            log.warning("Code collection is empty. Run code indexing first.")
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
                "file_ext": meta.get("file_ext", ""),
                "chunk_label": meta.get("chunk_label", ""),
                "symbols": meta.get("symbols", ""),
                "imports": meta.get("imports", ""),
                "score": 1 - dist,
            })

        return hits
