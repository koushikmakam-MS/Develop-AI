"""
Coder Agent - traces error code paths through the BMS service codebase.

Uses RAG over the indexed source code to:
  1. Find the code path that caused an error / handled a specific operation.
  2. Explain the flow (entry point → orchestrator → handler → data access).
  3. Identify potential root causes in the code.
  4. Detect cross-hive code boundaries (using statements, NuGet refs, API calls)
     and signal them for cross-hive resolution.

The agent pre-indexes the BMS repo for fast lookups and re-indexes daily
when new commits land.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

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
- Base your answers ONLY on the actual source code provided in the context.
- **DO NOT invent or fabricate code, class names, method names, file paths,
  REST endpoints, route attributes, or API contracts.** If the context
  doesn't contain a specific API/class/method, explicitly say so — do NOT
  make up plausible-looking code.
- If the code context doesn't contain the relevant files, say so and suggest
  which files/paths to look for.
- Be precise with file paths and class/method names — quote them exactly
  as they appear in the provided context.
- When explaining a flow, use a numbered list showing the call chain.
- **Highlight API details ONLY when present in the context**: Show REST API
  endpoints, HTTP methods, request/response contracts, controller action
  signatures, and route attributes — but ONLY if they actually appear in
  the provided code. Never invent endpoint URLs or route patterns.
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

        # Boundary detection: pattern → hive_name mapping
        # Set by BrainAgent/Router with set_boundary_map()
        self._boundary_map: Dict[str, str] = {}   # pattern → hive_name
        self._own_hive: Optional[str] = None

        # Callback to ask another hive's coder about a boundary
        self._boundary_callback: Optional[
            Callable[[str, str], Dict[str, Any]]
        ] = None

    def set_boundary_map(
        self,
        boundary_map: Dict[str, str],
        own_hive: str,
    ):
        """Set the boundary pattern → hive mapping.

        Parameters
        ----------
        boundary_map : dict
            Maps pattern strings (namespace prefixes, class names) to hive names.
            E.g. ``{"Microsoft.Azure.Common": "common", "CloudFMHelper": "common"}``
        own_hive : str
            This coder agent's owning hive name (to exclude self-matches).
        """
        self._boundary_map = boundary_map
        self._own_hive = own_hive
        log.info(
            "CoderAgent boundary map set: %d patterns, own_hive=%s",
            len(boundary_map), own_hive,
        )

    def set_boundary_callback(
        self,
        callback: Callable[[str, str], Dict[str, Any]],
    ):
        """Register a callback for resolving cross-hive code boundaries.

        Parameters
        ----------
        callback : callable(target_hive: str, question: str) -> dict
            Called when the coder detects a reference to another hive's code.
            Returns ``{"response": "...", "hive": "...", ...}``.
        """
        self._boundary_callback = callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, question: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Analyze a code-related question:
        1. Build multiple search queries from the question.
        2. Run them against the code index and merge / deduplicate hits.
        3. Assemble a context block that fits the model budget.
        4. Detect cross-hive code boundaries and resolve them.
        5. Ask the LLM to trace the code path and explain.
        """
        response, _ = self.analyze_with_boundaries(question, conversation_history)
        return response

    def analyze_simple(self, question: str, conversation_history: List[Dict] | None = None) -> str:
        """Analyze without boundary detection — used by cross-hive callbacks to prevent recursion."""
        conversation_history = conversation_history or []
        queries = self._build_search_queries(question)
        all_hits = self._multi_query_search(queries)

        if not all_hits:
            return "I don't have any indexed source code matching this query."

        context_block, seen_sources = self._build_context_block(all_hits)

        user_message = (
            f"## Source code context ({len(all_hits)} chunks from {len(seen_sources)} files):\n\n"
            f"{context_block}\n\n"
            f"## Files found: {', '.join(sorted(seen_sources))}\n\n"
            f"## User's question:\n{question}\n\n"
            f"Trace the code path and explain the flow."
        )

        return self.llm.generate(
            message=user_message,
            system=SYSTEM_PROMPT,
            history=conversation_history,
        )

    def analyze_with_boundaries(
        self,
        question: str,
        conversation_history: List[Dict] | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Like ``analyze()`` but also returns detected boundary crossings.

        Returns
        -------
        (response_text, boundaries) where boundaries is a list of dicts:
        ``[{"target_hive": "common", "pattern": "CloudFMHelper",
            "context": "...", "question": "...", "answer": "..."}]``
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
                "Make sure the repo is cloned (run `python run_sync.py` first).",
                [],
            )

        # Step 3: build context within the character budget
        context_block, seen_sources = self._build_context_block(all_hits)

        log.info(
            "Coder context: %d chunks, %d unique files, %dK chars",
            len(all_hits), len(seen_sources),
            len(context_block) // 1000,
        )

        # Step 4: detect cross-hive code boundaries
        # Run BOTH regex (for explicit code refs) AND LLM (for semantic
        # cross-service detection considering the user's question), then merge.
        boundaries = self._detect_boundaries(all_hits)
        if self._boundary_map:
            llm_boundaries = self._detect_boundaries_llm(all_hits, question)
            # Merge — dedupe by target_hive (keep first/regex one if dup)
            seen_targets = {b["target_hive"] for b in boundaries}
            for lb in llm_boundaries:
                if lb["target_hive"] not in seen_targets:
                    boundaries.append(lb)
                    seen_targets.add(lb["target_hive"])
        boundary_context = ""
        if boundaries:
            boundary_context = self._resolve_boundaries(boundaries, question)

        # Step 5: prompt the LLM
        user_message = (
            f"## Source code context from the BMS repository "
            f"({len(all_hits)} chunks from {len(seen_sources)} files):\n\n"
            f"{context_block}\n\n"
            f"## Files found: {', '.join(sorted(seen_sources))}\n\n"
        )

        if boundary_context:
            user_message += (
                f"## Cross-service code context (from other teams' codebases):\n\n"
                f"{boundary_context}\n\n"
            )

        user_message += (
            f"## User's question:\n{question}\n\n"
            f"Trace the code path and explain the flow using ONLY the code shown above. "
            f"Quote class names, method names, file paths, and API routes EXACTLY "
            f"as they appear — do NOT invent any. "
            f"If the context doesn't contain a specific API endpoint, controller, "
            f"or class needed to answer, say so explicitly rather than guessing. "
            f"If you have cross-service context, show the full end-to-end "
            f"chain including what happens when the code crosses into "
            f"another service's boundary. "
            f"If the user describes an error, identify where in the code "
            f"it originates."
        )

        response = self.llm.generate(
            message=user_message,
            system=SYSTEM_PROMPT,
            history=conversation_history,
        )

        # Step 6: grounding check — flag symbols not found in retrieved evidence
        evidence_corpus = context_block + "\n" + boundary_context + "\n" + " ".join(seen_sources)
        ungrounded = self._check_grounding(response, evidence_corpus)
        if ungrounded:
            log.warning(
                "Grounding check: %d symbol(s) in response not found in retrieved code: %s",
                len(ungrounded), ungrounded[:10],
            )
            warning = (
                "\n\n---\n"
                "> ⚠️ **Grounding warning** — the following symbols in the response above "
                "were NOT found in the indexed source code and may be fabricated. "
                "Treat them with caution and verify against the repo:\n"
                + "\n".join(f"> - `{s}`" for s in ungrounded[:15])
            )
            if len(ungrounded) > 15:
                warning += f"\n> - ... and {len(ungrounded) - 15} more"
            response = response + warning

        return response, boundaries

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
            # Also look for API endpoints / REST contracts
            queries.append(
                " ".join(operation_words)
                + " API endpoint route controller action"
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

    # ------------------------------------------------------------------
    # Cross-hive boundary detection
    # ------------------------------------------------------------------

    # TODO: Add multi-language boundary signals — Python (import, requests.get),
    #       Java (import, RestTemplate), Go (import, http.Get), etc. Currently C#-only.
    # Patterns that indicate cross-boundary references in C# code
    _BOUNDARY_SIGNALS = [
        # using statements:  using Microsoft.Azure.Common.CloudFMHelper;
        re.compile(r'using\s+([\w.]+)\s*;'),
        # Fully qualified type references:  Common.CloudUtilities.Validate(...)
        re.compile(r'([\w]+(?:\.[\w]+){2,})\s*[\.(]'),
        # NuGet PackageReference:  <PackageReference Include="RecoverySvcs.DataMover" />
        re.compile(r'PackageReference\s+Include="([\w.]+)"'),
        # Project reference paths
        re.compile(r'ProjectReference\s+Include="[^"]*[\\/]([\w.]+)[\\/]'),
        # REST / HTTP API calls:  HttpClient.GetAsync("api/dpp/..."), PostAsync, PutAsync, SendAsync
        re.compile(r'(?:GetAsync|PostAsync|PutAsync|DeleteAsync|SendAsync)\s*\(\s*["\']([^"\']+)["\']'),
        # URI / route patterns:  "api/dpp/...", "/api/backupmanagement/..."
        re.compile(r'["\'](/api/[\w/]+)["\']'),
        # Queue / Service Bus:  QueueClient("queueName"), SendMessageAsync, ServiceBusClient
        re.compile(r'QueueClient\s*\(\s*["\']([^"\']+)["\']'),
        re.compile(r'TopicClient\s*\(\s*["\']([^"\']+)["\']'),
        re.compile(r'ServiceBusSender\s*\(\s*["\']([^"\']+)["\']'),
        # Azure Storage Queue:  CloudQueue.GetQueueReference("name")
        re.compile(r'GetQueueReference\s*\(\s*["\']([^"\']+)["\']'),
        # gRPC / WCF channel calls:  ChannelFactory<ISomeService>, GrpcChannel.ForAddress
        re.compile(r'ChannelFactory<([\w.]+)>'),
        re.compile(r'GrpcChannel\.ForAddress\s*\(\s*["\']([^"\']+)["\']'),
        # Internal proxy / client class instantiation:  new DppInternalProxy(), new DataMoverClient()
        re.compile(r'new\s+([\w]+(?:Proxy|Client|Gateway|Dispatcher|Handler))\s*\('),
        # Interface references that indicate cross-service contracts
        re.compile(r':\s*(I[\w]+(?:Service|Contract|Interface|Proxy))\b'),
    ]

    # Generic namespaces / framework imports that appear in virtually every
    # file and do NOT indicate a meaningful cross-service dependency.
    _BOUNDARY_NOISE = {
        "system", "system.collections", "system.collections.generic",
        "system.linq", "system.text", "system.threading",
        "system.threading.tasks", "system.io", "system.net",
        "system.net.http", "system.componentmodel",
        "microsoft.extensions", "microsoft.extensions.logging",
        "microsoft.extensions.dependencyinjection",
        "microsoft.aspnetcore", "newtonsoft.json",
    }

    def _detect_boundaries(
        self, hits: List[Dict],
    ) -> List[Dict[str, str]]:
        """Scan code chunks for references to other hives' code boundaries.

        Returns a list of boundary dicts:
        ``[{"target_hive": "common", "pattern": "CloudFMHelper",
            "source_file": "DppFabric.cs", "context_line": "using ..."}]``
        """
        if not self._boundary_map:
            return []

        found: Dict[str, Dict[str, str]] = {}  # key → boundary dict (dedup)

        for hit in hits:
            text = hit.get("text", "")
            source = hit.get("source", "")
            imports = hit.get("imports", "")

            # Extract all potential namespace/class references
            # from both the chunk text AND the file-level imports metadata
            references: List[str] = []
            for pattern in self._BOUNDARY_SIGNALS:
                references.extend(pattern.findall(text))
            # Also check imports stored as metadata (using statements from file header)
            if imports:
                for imp in imports.split(", "):
                    imp = imp.strip()
                    if imp:
                        references.append(imp)

            # Match references against boundary map
            for ref in references:
                ref_lower = ref.lower()
                # Skip generic framework namespaces
                if ref_lower in self._BOUNDARY_NOISE:
                    continue
                # Also skip if any noise prefix matches the full ref
                if any(ref_lower.startswith(n + ".") for n in self._BOUNDARY_NOISE):
                    continue

                for boundary_pattern, target_hive in self._boundary_map.items():
                    # Skip self-references
                    if target_hive == self._own_hive:
                        continue
                    if boundary_pattern.lower() in ref_lower:
                        key = f"{target_hive}::{boundary_pattern}"
                        if key not in found:
                            # Find the context line
                            ctx_line = ""
                            for line in text.splitlines():
                                if boundary_pattern in line:
                                    ctx_line = line.strip()
                                    break
                            found[key] = {
                                "target_hive": target_hive,
                                "pattern": boundary_pattern,
                                "reference": ref,
                                "source_file": source,
                                "context_line": ctx_line,
                            }

        boundaries = list(found.values())
        if boundaries:
            log.info(
                "Detected %d cross-hive boundary crossing(s): %s",
                len(boundaries),
                [(b["target_hive"], b["pattern"]) for b in boundaries],
            )

        # Deduplicate: keep at most 1 boundary per target hive (most specific)
        seen_hives: set = set()
        deduped: List[Dict[str, str]] = []
        for b in boundaries:
            if b["target_hive"] not in seen_hives:
                seen_hives.add(b["target_hive"])
                deduped.append(b)
        return deduped[:5]  # Cap at 5 distinct target hives

    def _detect_boundaries_llm(
        self, hits: List[Dict], question: str
    ) -> List[Dict[str, str]]:
        """Use LLM to detect cross-service boundaries that regex missed.

        This catches implicit dependencies: REST calls via string URLs,
        queue names, comments mentioning other services, etc.
        Only called when regex detection found nothing.
        """
        if not self._boundary_map:
            return []

        # Build available hives list for the LLM
        available_hives = sorted(set(self._boundary_map.values()) - {self._own_hive})
        if not available_hives:
            return []

        # Compact code summary (first 3000 chars from top hits)
        code_sample = ""
        for hit in hits[:5]:
            code_sample += f"// File: {hit.get('source', '?')}\n"
            code_sample += hit.get("text", "")[:600] + "\n\n"
        code_sample = code_sample[:3000]

        prompt = (
            f"You are analyzing code from the '{self._own_hive}' service to find\n"
            f"OTHER services that should be consulted to fully answer the user's question.\n\n"
            f"Available services: {available_hives}\n\n"
            f"User's question: {question}\n\n"
            f"Code from '{self._own_hive}':\n```\n{code_sample}\n```\n\n"
            f"Identify OTHER services that are relevant. Include a service when:\n"
            f"  1. The code calls/depends on it (using statements, API calls, queue messages,\n"
            f"     proxy calls, interface references, HTTP requests).\n"
            f"  2. OR the user's question conceptually involves that service's domain\n"
            f"     (e.g. a question about 'data movement' relates to a datamover service\n"
            f"     even if the current code doesn't directly call it).\n\n"
            f"Respond with ONLY a JSON array. Each object must have:\n"
            f'  {{"target_hive": "<hive_name>", "reason": "<brief explanation>"}}\n\n'
            f"Include up to 5 services. If none are relevant, respond with: []\n"
            f"IMPORTANT: Only include services from the available list. No text outside the JSON."
        )

        try:
            result = self.llm.generate(
                message=prompt,
                system="You are a code analyst. Respond ONLY with valid JSON.",
                history=[],
            )
            # Parse JSON response
            import json
            # Strip markdown code fences if present
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result
                result = result.rsplit("```", 1)[0]
            result = result.strip()

            detected = json.loads(result)
            if not isinstance(detected, list):
                return []

            boundaries = []
            for item in detected[:5]:  # Cap at 5 from LLM
                target = item.get("target_hive", "")
                reason = item.get("reason", "")
                if target in available_hives:
                    boundaries.append({
                        "target_hive": target,
                        "pattern": f"[LLM] {reason[:50]}",
                        "reference": reason,
                        "source_file": hits[0].get("source", "") if hits else "",
                        "context_line": f"LLM-detected: {reason}",
                    })
            if boundaries:
                log.info(
                    "LLM detected %d cross-hive boundary(s): %s",
                    len(boundaries),
                    [(b["target_hive"], b["pattern"]) for b in boundaries],
                )
            return boundaries

        except Exception as e:
            log.debug("LLM boundary detection failed: %s", e)
            return []

    def _resolve_boundaries(
        self,
        boundaries: List[Dict[str, str]],
        original_question: str,
    ) -> str:
        """Resolve detected boundaries by calling target hive coders.

        First asks the LLM which boundaries are actually critical for the
        user's question, then only resolves those.

        Returns a formatted context string with answers from other hives.
        """
        if not self._boundary_callback:
            log.info("No boundary callback registered — skipping resolution.")
            return ""

        # ── LLM relevance gate ──────────────────────────────────────
        boundaries = self._filter_critical_boundaries(boundaries, original_question)
        if not boundaries:
            log.info("LLM relevance gate: no boundaries are critical — skipping.")
            return ""

        sections: List[str] = []
        for boundary in boundaries:
            target = boundary["target_hive"]
            pattern = boundary["pattern"]
            ref = boundary.get("reference", pattern)
            source = boundary.get("source_file", "unknown")
            ctx_line = boundary.get("context_line", "")

            # Build a targeted question for the other hive's coder
            sub_question = (
                f"In the context of '{original_question}': "
                f"Explain what `{ref}` does. "
                f"It is referenced from `{source}` "
            )
            if ctx_line:
                sub_question += f"in this line: `{ctx_line}`. "
            sub_question += (
                "Trace the code path for this class/method. "
                "What does it do internally?"
            )

            log.info(
                "Boundary → %s: asking about %s (from %s)",
                target, pattern, source,
            )

            try:
                result = self._boundary_callback(target, sub_question)
                answer = result.get("response", "")
                if answer:
                    boundary["question"] = sub_question
                    boundary["answer"] = answer[:3000]
                    sections.append(
                        f"### From {target.upper()} (boundary: `{pattern}`)\n"
                        f"*Called from:* `{source}`\n"
                        f"*Reference:* `{ref}`\n\n"
                        f"{answer[:3000]}\n"
                    )
            except Exception as e:
                log.warning("Boundary resolution for %s failed: %s", target, e)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # LLM relevance gate for boundary crossings
    # ------------------------------------------------------------------

    _BOUNDARY_RELEVANCE_PROMPT = (
        "You are deciding which cross-service code references are CRITICAL\n"
        "to answer the user's question.\n\n"
        "User's question: \"{question}\"\n\n"
        "Detected cross-service references:\n{boundary_list}\n\n"
        "For each reference, decide: is understanding this reference's\n"
        "implementation ESSENTIAL to properly answer the question?\n\n"
        "A reference is CRITICAL when:\n"
        "- It is a domain-specific class/service that performs key logic\n"
        "  for the operation the user asked about.\n"
        "- Without it, the answer would be incomplete or misleading.\n\n"
        "A reference is NOT critical when:\n"
        "- It is a generic utility, logging, validation, or infrastructure\n"
        "  namespace (e.g. common helpers, base classes, config readers).\n"
        "- The user didn't ask about that part of the system.\n\n"
        "Respond with ONLY the indices (0-based) of the CRITICAL references,\n"
        "comma-separated. If none are critical, respond with: NONE\n\n"
        "Example: 0,2\n"
        "Example: NONE"
    )

    def _filter_critical_boundaries(
        self,
        boundaries: List[Dict[str, str]],
        question: str,
    ) -> List[Dict[str, str]]:
        """Ask the LLM which detected boundaries are critical for the question."""
        if not boundaries:
            return []

        # Build a numbered list for the LLM
        lines = []
        for i, b in enumerate(boundaries):
            lines.append(
                f"[{i}] target_hive={b['target_hive']}, "
                f"reference={b.get('reference', b['pattern'])}, "
                f"source_file={b.get('source_file', '?')}, "
                f"context={b.get('context_line', '')[:120]}"
            )

        prompt = self._BOUNDARY_RELEVANCE_PROMPT.format(
            question=question,
            boundary_list="\n".join(lines),
        )

        try:
            result = self.llm.generate(
                message=prompt,
                system="Be strict. Only mark references as critical if they are essential.",
                history=[],
            ).strip()

            if result.upper().startswith("NONE"):
                log.info("LLM boundary gate: none are critical.")
                return []

            # Parse comma-separated indices
            indices = set()
            for token in result.replace(" ", "").split(","):
                token = token.strip()
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < len(boundaries):
                        indices.add(idx)

            critical = [boundaries[i] for i in sorted(indices)]
            log.info(
                "LLM boundary gate: %d/%d critical: %s",
                len(critical), len(boundaries),
                [(b["target_hive"], b.get("reference", b["pattern"])[:40]) for b in critical],
            )
            return critical

        except Exception as e:
            log.warning("LLM boundary relevance check failed: %s — keeping all.", e)
            return boundaries  # Fallback: keep all if LLM fails

    # ------------------------------------------------------------------
    # Grounding check — flag fabricated symbols
    # ------------------------------------------------------------------

    # Common framework/language identifiers that should be exempt from
    # the grounding check (they're standard, not "fabricated" even if
    # they don't appear in the indexed code).
    _GROUNDING_EXEMPT = {
        # C# / .NET keywords & common types
        "string", "int", "bool", "void", "var", "async", "await",
        "task", "list", "dictionary", "object", "null", "true", "false",
        "public", "private", "protected", "internal", "static", "class",
        "interface", "namespace", "using", "return", "new", "this",
        "ienumerable", "icollection", "ilist", "idictionary", "ireadonly",
        "exception", "argumentexception", "argumentnullexception",
        "invalidoperationexception", "notsupportedexception",
        "datetime", "datetimeoffset", "timespan", "guid", "uri",
        "httpclient", "httpresponsemessage", "httprequestmessage",
        "httpget", "httppost", "httpput", "httpdelete", "httppatch",
        "fromBody", "frombody", "fromquery", "fromroute", "fromheader",
        "ok", "badrequest", "notfound", "actionresult", "iactionresult",
        "controllerbase", "controller", "route", "apicontroller",
        "console", "writeline", "system", "microsoft", "azure",
        "logger", "ilogger", "log", "info", "error", "warning", "debug",
        # Generic terms
        "request", "response", "result", "data", "service", "client",
        "manager", "handler", "controller", "context", "options",
        "configuration", "settings", "model", "view", "viewmodel",
    }

    # Match symbols inside fenced code blocks
    _CODE_FENCE_RE = re.compile(r"```[\w]*\n(.*?)\n```", re.DOTALL)
    # PascalCase identifiers (likely class names)
    _PASCAL_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{3,})\b")
    # Method calls: Foo.Bar( or .Bar(
    _METHOD_RE = re.compile(r"\.([A-Z][a-zA-Z0-9]{2,})\s*\(")
    # Route strings:  "/api/..." or '/api/...'
    _ROUTE_RE = re.compile(r"""['"](/api/[\w/{}.\-]+)['"]""")
    # File paths (.cs, .py, etc.) — heuristic
    _PATH_RE = re.compile(r"\b([\w/\\.\-]+\.(?:cs|py|java|ts|js|go))\b")

    def _check_grounding(
        self,
        response: str,
        evidence_corpus: str,
    ) -> List[str]:
        """Find code symbols in the response that don't appear in the
        retrieved evidence. Returns a list of unique ungrounded symbols.

        Only checks symbols inside fenced code blocks (where hallucinated
        APIs typically appear) — prose explanations are not checked since
        the LLM may legitimately paraphrase concepts.
        """
        if not response or not evidence_corpus:
            return []

        evidence_lower = evidence_corpus.lower()

        # Extract all fenced code blocks
        code_blocks = self._CODE_FENCE_RE.findall(response)
        if not code_blocks:
            return []

        symbols: set = set()
        for block in code_blocks:
            # PascalCase class/type names
            for sym in self._PASCAL_RE.findall(block):
                symbols.add(sym)
            # Method calls
            for sym in self._METHOD_RE.findall(block):
                symbols.add(sym)
            # Route strings
            for sym in self._ROUTE_RE.findall(block):
                symbols.add(sym)
            # File paths
            for sym in self._PATH_RE.findall(block):
                symbols.add(sym)

        ungrounded: List[str] = []
        for sym in symbols:
            sym_lower = sym.lower()
            # Skip exempt framework identifiers
            if sym_lower in self._GROUNDING_EXEMPT:
                continue
            # Skip very short symbols (false positive risk)
            if len(sym) < 4:
                continue
            # Check if symbol appears anywhere in the evidence
            if sym_lower not in evidence_lower:
                ungrounded.append(sym)

        # Sort for stable output, dedupe (already a set)
        return sorted(set(ungrounded))