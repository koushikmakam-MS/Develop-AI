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
include a Mermaid diagram to visualize it. Use the appropriate diagram type:
- `flowchart TD` for code execution paths, branching logic, and error handling flows
- `sequenceDiagram` for call chains between classes/services (Controller → Orchestrator → Handler → DB)
- `classDiagram` for class hierarchies and interface implementations
- `graph LR` for component/module dependency diagrams

Wrap diagrams in a ```mermaid code fence. Keep them focused (under 20 nodes).
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

        log.info(
            "CoderAgent initialized (code collection: %d chunks)",
            self.code_indexer.collection.count(),
        )

    def analyze(self, question: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Analyze a code-related question:
        1. Search the code index for relevant source files.
        2. Build a context-augmented prompt with code snippets.
        3. Ask LLM to trace the code path and explain.
        """
        # Build a rich search query
        search_query = self._build_search_query(question)

        # Search code index
        hits = self.code_indexer.search(search_query, top_k=10)

        if not hits:
            return (
                "I don't have any indexed source code yet. "
                "Please run `python run_code_index.py` to index the BMS repository first.\n\n"
                "Make sure the repo is cloned (run `python run_sync.py` first)."
            )

        # Build context block from code hits
        context_parts = []
        seen_sources = set()
        for i, hit in enumerate(hits, 1):
            source = hit["source"]
            label = hit.get("chunk_label", "")
            ext = hit.get("file_ext", "")

            header = f"--- Code [{i}]: {source}"
            if label:
                header += f" (near: {label})"
            header += f" (relevance: {hit['score']:.2f}) ---"

            context_parts.append(f"{header}\n```{ext.lstrip('.')}\n{hit['text']}\n```\n")
            seen_sources.add(source)

        context_block = "\n".join(context_parts)

        # Build the user message
        user_message = (
            f"## Source code context from the BMS repository:\n\n"
            f"{context_block}\n\n"
            f"## Files found: {', '.join(sorted(seen_sources))}\n\n"
            f"## User's question:\n{question}\n\n"
            f"Trace the code path and explain the flow. "
            f"If the user describes an error, identify where in the code it originates."
        )

        response = self.llm.generate(
            message=user_message,
            system=SYSTEM_PROMPT,
            history=conversation_history,
        )

        return response

    def _build_search_query(self, question: str) -> str:
        """
        Enhance the search query to find more relevant code.
        Extracts error names, class references, operation names, etc.
        """
        # Start with the raw question
        parts = [question]

        # Extract potential class/method names (PascalCase or camelCase words)
        import re
        pascal_words = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", question)
        if pascal_words:
            parts.extend(pascal_words)

        # Extract quoted strings (often error messages or operation names)
        quoted = re.findall(r'"([^"]+)"', question)
        parts.extend(quoted)
        quoted = re.findall(r"'([^']+)'", question)
        parts.extend(quoted)

        return " ".join(parts)
