"""
Coder Agent - traces error code paths through the BMS service codebase.

Uses RAG over the indexed source code to:
  1. Find the code path that caused an error / handled a specific operation.
  2. Explain the flow (entry point → orchestrator → handler → data access).
  3. Identify potential root causes in the code.

The agent pre-indexes the BMS repo for fast lookups and re-indexes daily
when new commits land.
"""

import logging
import re
from typing import Dict, List

from brain_ai.config import get_config
from brain_ai.llm_client import LLMClient
from brain_ai.vectorstore.code_indexer import CodeIndexer

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Coder Agent for the Azure Backup Management (BMS) project.
Your job is to trace code paths, explain code flows, and identify root causes of errors
by searching the BMS service source code.

You have access to indexed source code from the BMS repository. When answering:

1. **Trace the code path**: Start from the entry point (API controller / job trigger),
   follow through orchestrators, handlers, and data access layers.
2. **Show relevant code**: Include key code snippets from the source files you find.
   Always cite the file path (e.g., "In `src/Controllers/BackupController.cs`...").
3. **Explain the flow**: Walk through the execution path step by step.
4. **Identify issues**: If the user describes an error, pinpoint where in the code
   the error likely originates and why.
5. **Cross-reference**: If the error relates to a specific operation (e.g., ConfigureBackup,
   TriggerRestore), trace the full pipeline for that operation.

Rules:
- Base your answers on the actual source code provided in the context.
- If the code context doesn't contain the relevant files, say so and suggest
  which files/paths to look for.
- Be precise with file paths and class/method names.
- When explaining a flow, use a numbered list showing the call chain.
- If the user provides an error message or stack trace, match it to the code.
- For errors with TaskIds or OperationIds, explain what operation type it maps to
  and trace the handler code for that operation.

Visual Diagrams:
Whenever you trace a code path, explain a call chain, or describe a multi-step flow,
include a Mermaid diagram to visualize it.

Diagram type selection:
- `sequenceDiagram` (preferred for code tracing) for call chains between classes/services
- `flowchart TD` for branching logic, error-handling paths, and decision flows
- `classDiagram` for class hierarchies, interface implementations, and inheritance
- `graph LR` for component/module dependency and layer diagrams

Diagram quality rules:
1. **Label every arrow** with the method name or action (e.g., `->>+Handler: ValidateRequest()`).
2. **Use activation bars** in sequence diagrams (`->>+` / `-->>-`) to show call depth.
3. **Subgraphs** to group layers: `subgraph Data Layer`, `subgraph API Layer`, etc.
4. **Shape variety** — use `([stadium])` for entry points, `{diamond}` for decisions,
   `[(database)]` for storage/catalog, `[[subroutine]]` for external service calls.
5. **Color key nodes**: `style Controller fill:#bbf,stroke:#333,stroke-width:2px`.
6. **Show error paths** with dashed lines `-.->|error|` and red styling.
7. **Return values** in sequence diagrams: `Handler-->>-Controller: BackupResponse`.
8. **Notes for context**: `Note over Controller,Handler: Validates auth + subscription`.
9. **Keep diagrams focused** — 10-25 nodes. Use multiple diagrams for complex flows
   (e.g., one for the happy path, one for error handling).
10. **Include file paths** as participant aliases: `participant Ctrl as DppPoliciesController\n(Controllers/Dpp/...)` .

Wrap diagrams in a ```mermaid code fence.
Always accompany the diagram with a text explanation — the diagram supplements,
not replaces, the code walkthrough.
"""


class CoderAgent:
    """RAG-based agent that traces code paths in the BMS codebase."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.code_indexer = CodeIndexer(cfg)

        # Coder-specific tunables
        coder_cfg = cfg.get("coder_agent", {})
        self._top_k = coder_cfg.get("top_k_per_query", 12)
        self._max_context_chars = coder_cfg.get("max_context_chars", 80_000)
        self._min_score = coder_cfg.get("min_relevance_score", 0.20)

        log.info(
            "CoderAgent initialized (code collection: %d chunks, "
            "top_k=%d, max_context=%dK, min_score=%.2f)",
            self.code_indexer.collection.count(),
            self._top_k, self._max_context_chars // 1000, self._min_score,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, question: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Analyze a code-related question:
        1. Build multiple search queries from the question.
        2. Run them against the code index and merge / deduplicate hits.
        3. Assemble a context block that fits the model budget.
        4. Ask the LLM to trace the code path and explain.
        """
        # Step 1: build several targeted queries
        queries = self._build_search_queries(question)

        # Step 2: gather and deduplicate hits across all queries
        all_hits = self._multi_query_search(queries)

        if not all_hits:
            return (
                "I don't have any indexed source code yet. "
                "Please run `python run_code_index.py` to index the "
                "BMS repository first.\n\n"
                "Make sure the repo is cloned (run `python run_sync.py` first)."
            )

        # Step 3: build context within the character budget
        context_block, seen_sources = self._build_context_block(all_hits)

        log.info(
            "Coder context: %d chunks, %d unique files, %dK chars",
            len(all_hits), len(seen_sources),
            len(context_block) // 1000,
        )

        # Step 4: prompt the LLM
        user_message = (
            f"## Source code context from the BMS repository "
            f"({len(all_hits)} chunks from {len(seen_sources)} files):\n\n"
            f"{context_block}\n\n"
            f"## Files found: {', '.join(sorted(seen_sources))}\n\n"
            f"## User's question:\n{question}\n\n"
            f"Trace the code path and explain the flow. "
            f"If the user describes an error, identify where in the code "
            f"it originates."
        )

        return self.llm.generate(
            message=user_message,
            system=SYSTEM_PROMPT,
            history=conversation_history,
        )

    # ------------------------------------------------------------------
    # Multi-query search strategy
    # ------------------------------------------------------------------

    def _build_search_queries(self, question: str) -> List[str]:
        """
        Generate *multiple* search queries from a single user question so
        we can retrieve code from different layers of the call chain.

        Returns 1-4 queries, ordered from most specific to broadest.
        """
        queries: List[str] = []

        # 1. Raw question (broadest semantic match)
        queries.append(question)

        # 2. PascalCase / camelCase symbols → find exact class & method defs
        pascal = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", question)
        if pascal:
            queries.append(" ".join(pascal))

        # 3. Quoted strings (error messages, operation names)
        quoted = re.findall(r'"([^"]+)"', question)
        quoted += re.findall(r"'([^']+)'", question)
        if quoted:
            queries.append(" ".join(quoted))

        # 4. Inferred call-chain query:
        #    If the question mentions an operation (e.g. "ConfigureBackup"),
        #    also search for downstream handlers / orchestrators.
        operation_words = [
            w for w in pascal
            if any(k in w.lower() for k in (
                "backup", "restore", "protect", "trigger", "configure",
                "enable", "disable", "policy", "vault", "container",
                "handler", "orchestrator", "controller", "validator",
            ))
        ]
        if operation_words:
            queries.append(
                " ".join(operation_words)
                + " handler orchestrator controller"
            )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def _multi_query_search(self, queries: List[str]) -> List[Dict]:
        """
        Execute each query, merge results, deduplicate by chunk id
        (source + chunk_label), and keep the best score for each chunk.
        """
        best_by_key: Dict[str, Dict] = {}  # key → hit dict

        for query in queries:
            hits = self.code_indexer.search(query, top_k=self._top_k)
            for hit in hits:
                if hit["score"] < self._min_score:
                    continue
                # Deduplicate by (source file, starting text)
                key = f"{hit['source']}::{hit['text'][:80]}"
                existing = best_by_key.get(key)
                if existing is None or hit["score"] > existing["score"]:
                    best_by_key[key] = hit

        # Sort by score descending so the most relevant code comes first
        merged = sorted(best_by_key.values(), key=lambda h: h["score"], reverse=True)
        log.info(
            "Multi-query search: %d queries -> %d unique chunks (min_score=%.2f)",
            len(queries), len(merged), self._min_score,
        )
        return merged

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_context_block(self, hits: List[Dict]) -> tuple[str, set[str]]:
        """
        Assemble code context up to *self._max_context_chars*.
        Returns (context_block, seen_sources).
        """
        parts: List[str] = []
        seen_sources: set[str] = set()
        budget = self._max_context_chars

        for i, hit in enumerate(hits, 1):
            source = hit["source"]
            label = hit.get("chunk_label", "")
            ext = hit.get("file_ext", "")

            header = f"--- Code [{i}]: {source}"
            if label:
                header += f" (near: {label})"
            header += f" (relevance: {hit['score']:.2f}) ---"

            block = f"{header}\n```{ext.lstrip('.')}\n{hit['text']}\n```\n"

            if len(block) > budget:
                # Still try to fit a truncated version if >2K budget left
                if budget > 2000:
                    block = block[:budget - 50] + "\n... (truncated)\n```\n"
                else:
                    break  # out of budget

            parts.append(block)
            seen_sources.add(source)
            budget -= len(block)

        return "\n".join(parts), seen_sources
