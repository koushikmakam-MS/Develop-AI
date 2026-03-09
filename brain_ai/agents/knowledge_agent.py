"""
Knowledge Agent - answers project-related questions using RAG
over the indexed markdown documentation.
"""

import logging
from typing import Dict, List

from brain_ai.config import get_config
from brain_ai.llm_client import LLMClient
from brain_ai.vectorstore.indexer import DocIndexer

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Knowledge Agent for the Azure Backup Management project.
Your job is to answer questions about the project using the provided documentation context.

Rules:
- Base your answers ONLY on the provided context from the project docs.
- If the context doesn't contain enough information, say so clearly.
- Reference the source document when possible (e.g., "According to <filename>...").
- Be concise and technical.
- If asked about debugging or Kusto queries, mention that the Debug Agent can help with that.

Visual Diagrams:
Whenever your answer involves a workflow, architecture, request flow, state machine,
or multi-step process, include a Mermaid diagram to illustrate it.

Diagram type selection:
- `flowchart TD` for workflows, decision trees, branching logic, and error-handling paths
- `sequenceDiagram` for request/response flows, API call chains, and multi-service interactions
- `stateDiagram-v2` for state machines, lifecycle transitions, and status progressions
- `graph LR` for component relationships, architecture overviews, and dependency maps
- `classDiagram` for class hierarchies and interface implementations (when relevant)

Diagram quality rules:
1. **Label every edge** — use `-->|label|` or `->>` with notes; never leave arrows unlabelled.
2. **Use subgraphs** to group related nodes (e.g., `subgraph API Layer`).
3. **Shape variety** — use `([stadium])` for start/end, `{diamond}` for decisions,
   `[(database)]` for storage, `[[subroutine]]` for external calls.
4. **Color / style** key nodes: `style NodeId fill:#f9f,stroke:#333,stroke-width:2px`.
5. **Keep diagrams focused** — 8-20 nodes. Split into multiple diagrams if needed.
6. **Add notes** in sequence diagrams: `Note over A,B: explanation`.
7. **Show error paths** — use dashed lines `-.->` for error/fallback flows.
8. **Include return values** where applicable (e.g., `B-->>A: 200 OK`).

Wrap diagrams in a ```mermaid code fence.
Always provide a text explanation alongside the diagram — the diagram supplements,
not replaces, the written answer.
"""


class KnowledgeAgent:
    """RAG-based Q&A agent over project documentation."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.indexer = DocIndexer(cfg)

    def answer(self, question: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Answer a question using RAG.
        1. Search the vector store for relevant chunks.
        2. Build a context-augmented prompt.
        3. Send to LLM and return the response.
        """
        response, _ = self.answer_with_confidence(question, conversation_history)
        return response

    def answer_with_confidence(
        self, question: str, conversation_history: List[Dict] | None = None
    ) -> tuple[str, float]:
        """
        Same as `answer()` but also returns the best relevance score (0.0–1.0)
        so the caller can decide whether the knowledge base had a good match.
        Returns (response_text, best_score).
        """
        # Retrieve relevant docs
        hits = self.indexer.search(question, top_k=6)

        if not hits:
            return (
                "I don't have any indexed documentation yet. "
                "Please run `python run_sync.py` then `python run_index.py` first.",
                0.0,
            )

        best_score = max(h["score"] for h in hits)
        log.info("Knowledge search — best score: %.3f for %d hits", best_score, len(hits))

        # Build context block
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"--- Document {i}: {hit['source']} (relevance: {hit['score']:.2f}) ---\n"
                f"{hit['text']}\n"
            )
        context_block = "\n".join(context_parts)

        user_message = (
            f"## Context from project documentation:\n\n{context_block}\n\n"
            f"## Question:\n{question}"
        )

        log.debug("Sending to LLM with %d context chunks.", len(hits))

        response = self.llm.generate(
            message=user_message,
            system=SYSTEM_PROMPT,
            history=conversation_history,
        )
        return response, best_score
