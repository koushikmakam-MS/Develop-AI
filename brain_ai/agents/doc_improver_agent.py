"""
Doc Improver Agent — Background workflow that reads source code from the repo
and ChromaDB vector store, compares against existing documentation, identifies
gaps / inaccuracies, and creates a PR with improved docs.

Architecture:
  1. Load existing docs (from docs/agentKT/) — these are the "reference"
  2. Load Telemetry_And_Logging_Reference.md for KQL patterns
  3. For each feature doc, use gpt-5.1-codex-mini (code-reader) to:
     a. Search the code index for relevant source code
     b. Compare code reality vs doc claims
     c. Identify missing data, incorrect information, missing debug patterns
     d. Generate an improved doc (or a new doc for undocumented features)
  4. After all docs are processed, diff against originals
  5. If meaningful changes found → create one PR with all improvements

Rules:
  - BackupMgmt_Architecture_Memory.md is NEVER modified
  - Retains DPP/ and RSV/ folder structure
  - Each doc follows a standard template
  - Includes debugging patterns and KQL query references from Telemetry doc
  - Only creates a PR if substantive new content is discovered
"""

import difflib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain_ai.config import get_config
from brain_ai.code_reader_llm import CodeReaderLLM
from brain_ai.llm_client import LLMClient
from brain_ai.vectorstore.indexer import DocIndexer
from brain_ai.vectorstore.code_indexer import CodeIndexer
from brain_ai.sync.devops_pr import AzureDevOpsPR

log = logging.getLogger(__name__)

# ── Standard doc template that the agent enforces ────────────────────

DOC_TEMPLATE_DESCRIPTION = """
Every feature doc MUST follow this standard template structure:

# Feature: <Feature Name>

> **Purpose**: One-line description.

---

## 1. API Endpoints
Table of HTTP method, route, FM constant.

## 2. Request/Response Flow
Step-by-step code path from controller → business logic → catalog/fabric.
Include class names and method names from the source code.

## 3. Business Logic & Validation
Key validation rules, state checks, error conditions.

## 4. State Machine / Lifecycle
States, transitions, guard conditions (if applicable).

## 5. Catalog / Database Operations
What gets written/read, stored procedures or EF operations.

## 6. Telemetry & Logging
Which OpStats are logged, activity stats, trace events.
Map to Kusto table columns.

## 7. Debugging Patterns & KQL Queries
Concrete KQL queries for common debug scenarios specific to this feature.
Reference the Telemetry_And_Logging_Reference.md for table schemas.
Include:
  - How to find failed operations for this feature
  - How to trace the full request flow
  - How to identify error sources
  - Performance investigation queries

## 8. Error Handling
Common error codes, retry logic, failure modes.

## 9. Related Features
Cross-references to other feature docs.
"""

# ── System prompts ───────────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are a senior software engineer analyzing Azure Backup Management source code.

Your job: Compare the source code against an existing documentation file and identify:
1. **Missing information** — code paths, classes, methods, validations not documented
2. **Incorrect information** — doc claims that contradict the actual code
3. **Missing debugging patterns** — KQL queries that would help debug this feature
4. **Missing telemetry references** — OpStats, activity logs, trace events in the code but not in the doc

You will be given:
- The current documentation content
- Relevant source code snippets from the codebase
- The telemetry reference document (for KQL patterns)

Output a JSON analysis:
{
  "has_changes": true/false,
  "missing_items": ["list of missing things"],
  "incorrect_items": ["list of incorrect things"],
  "missing_kql_patterns": ["KQL queries that should be added"],
  "summary": "one-line summary of changes needed",
  "severity": "high/medium/low"
}

Be thorough but precise. Only flag real issues backed by code evidence.
Do NOT flag stylistic preferences or minor formatting issues.
"""

IMPROVE_DOC_SYSTEM_PROMPT = """You are a technical writer for Azure Backup Management documentation.

You are given:
1. An existing documentation file
2. An analysis of what needs improving (missing items, incorrect items, KQL patterns)
3. The source code evidence supporting the changes
4. The telemetry reference for KQL query patterns

Your job: Produce an IMPROVED version of the document that:
- Fixes all identified inaccuracies based on the source code
- Adds missing information discovered in the code
- Adds debugging patterns with concrete KQL queries
- Follows the standard template structure
- Preserves all existing CORRECT information
- Integrates new content naturally into existing sections
- References source file paths where helpful

Rules:
- Return ONLY the full improved markdown document — no code fences, no preamble
- Keep the same file's overall topic scope — don't add unrelated features
- Every KQL query MUST reference correct Kusto table names:
  OperationStatsLocalAllClusters, TraceLogMessageAllClusters, ArtifactSnapshotAll
- Include FM constant names (e.g., DppWebUpsertBackupInstance) in KQL where applicable
"""

NEW_DOC_SYSTEM_PROMPT = """You are a technical writer creating NEW documentation for an Azure Backup Management feature.

You are given:
1. Source code snippets for a feature that currently has NO documentation
2. The telemetry reference for KQL query patterns
3. The standard template structure to follow

Create a comprehensive feature documentation file following the standard template.

Rules:
- Follow the standard template exactly (all 9 sections)
- Include concrete KQL debugging queries using correct table names
- Reference actual class names, method names, and file paths from the code
- Return ONLY the markdown document — no code fences, no preamble
"""

DISCOVERY_SYSTEM_PROMPT = """You are analyzing an Azure Backup Management codebase to discover features
that may be undocumented or under-documented.

You will be given:
1. A list of existing documented features (file names)
2. Source code snippets from the codebase showing controllers, APIs, and key classes

Identify features or major code areas that exist in the code but have NO corresponding
documentation file. Only flag substantial features (not utility classes or helpers).

Output JSON:
{
  "undocumented_features": [
    {
      "feature_name": "descriptive name",
      "evidence": "controller/class names found in code",
      "suggested_filename": "Feature_Name.md",
      "folder": "DPP or RSV",
      "search_queries": ["queries to find more code for this feature"]
    }
  ]
}
"""

BOOTSTRAP_CODEBASE_MAP_PROMPT = """You are a senior Azure Backup Management engineer.

You are given source code from a codebase that has NO documentation at all.
Your job is to analyze the code and produce a comprehensive map of all major features
that need to be documented.

For this codebase, features typically map to:
- Controllers (DppWeb*, BMS*Controller)
- API surfaces (REST endpoints for backup instances, policies, jobs, etc.)
- Major workflows (backup, restore, cross-region restore, data move)
- Management operations (suspend/resume, stop/delete, resource guards)

Analyze the code samples and classify each feature into either:
- **DPP** (Data Protection Platform — newer stack using DppWeb* controllers)
- **RSV** (Recovery Services Vault — older stack using BMS*Controller / *ProtectedItems*)

Output JSON:
{
  "architecture_summary": "2-3 paragraph high-level description of the codebase architecture",
  "features": [
    {
      "feature_name": "descriptive name",
      "evidence": "controller/class names found in code",
      "suggested_filename": "Feature_Name.md",
      "folder": "DPP or RSV",
      "search_queries": ["queries to find more code for this feature"],
      "priority": "high/medium/low"
    }
  ],
  "telemetry_patterns": [
    "List of telemetry table names, OpStats patterns, and KQL query patterns found in the code"
  ]
}

Be exhaustive — list EVERY feature you can identify. Mark priority 'high' for core
operations (backup/restore/policy), 'medium' for management operations, 'low' for
utility/config features.
"""

BOOTSTRAP_ARCHITECTURE_PROMPT = """You are a senior engineer creating the ARCHITECTURE reference document
for an Azure Backup Management codebase that has never been documented.

You are given:
1. A high-level architecture summary derived from the code
2. A list of all identified features
3. Source code samples showing the overall structure

Create a comprehensive architecture document covering:
- System Overview (what the codebase does, DPP vs RSV stacks)
- High-Level Architecture (layers, services, data flow)
- Key Abstractions (controllers, validators, catalogs, fabric adapters)
- Feature Map (table of all features with their folder/file)
- Common Patterns (validation flow, error handling, state machines)
- Deployment & Configuration context

Rules:
- This document serves as the TOP-LEVEL reference that all feature docs link back to
- Include a table of contents
- Reference actual namespace/class names from the code
- Return ONLY the markdown document — no code fences, no preamble
"""

BOOTSTRAP_TELEMETRY_PROMPT = """You are creating the TELEMETRY & LOGGING reference document
for an Azure Backup Management codebase.

You are given:
1. Source code samples showing telemetry, logging, and tracing patterns
2. Telemetry patterns identified during code analysis

Create a comprehensive telemetry reference covering:
- Kusto Tables & Schemas (with actual column names from the code)
- OpStats Logging (patterns, FM constant mapping)
- Activity Stats & Trace Logging
- Common KQL Debugging Queries (parameterized, with explanations)
- Error Code Mapping
- Performance Investigation Patterns

Rules:
- Include actual Kusto table names: OperationStatsLocalAllClusters,
  TraceLogMessageAllClusters, ArtifactSnapshotAll, etc.
- Include parameterized KQL queries that other docs can reference
- Return ONLY the markdown document — no code fences, no preamble
"""


class DocImproverAgent:
    """
    Background agent that reads code + vector DB, compares against docs,
    and produces improved documentation with a PR.
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg

        imp_cfg = cfg.get("doc_improver", {})
        self.max_iterations = imp_cfg.get("max_iterations", 3)
        self.protected_docs = set(imp_cfg.get("protected_docs", ["BackupMgmt_Architecture_Memory.md"]))
        self.branch_prefix = imp_cfg.get("branch_prefix", "BCDR-devai/doc-improvement")
        self.min_diff_lines = imp_cfg.get("min_diff_lines", 10)
        self.code_folders = imp_cfg.get("code_folders", [])
        self.max_new_docs_per_cycle = imp_cfg.get("max_new_docs_per_cycle", 15)

        # Paths
        self.docs_dir = Path(cfg["paths"]["docs_dir"]).resolve()
        self.repo_clone_dir = Path(cfg["paths"]["repo_clone_dir"]).resolve()

        # Track original content before this cycle (for accurate diffing)
        # key = file_path (str), value = original content or None (new file)
        self._original_content: Dict[str, Optional[str]] = {}

        # LLM clients
        self.code_reader = CodeReaderLLM(cfg)  # gpt-5.1-codex-mini for code reading
        self.writer_llm = LLMClient(cfg)       # gpt-4o-mini for doc writing

        # Vector store
        self.doc_indexer = DocIndexer(cfg)
        self.code_indexer = CodeIndexer(cfg)

        # PR helper
        self.pr_helper = AzureDevOpsPR(cfg)

        # Telemetry reference (loaded once, used in every doc improvement)
        self._telemetry_ref: Optional[str] = None

        log.info("DocImproverAgent initialized (max_iterations=%d, protected=%s)",
                 self.max_iterations, self.protected_docs)

    # ── Public API ───────────────────────────────────────────────────

    def run_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run the full doc improvement cycle:
          1. Load all existing docs
          2. For each doc: analyze code vs doc → improve if needed
          3. Discover undocumented features → create new docs
          4. Iterate up to max_iterations (converge when no changes)
          5. If changes found → create one PR

        Returns a summary dict.
        """
        log.info("=" * 60)
        log.info("DOC IMPROVEMENT CYCLE STARTING")
        log.info("=" * 60)

        # Snapshot existing doc contents BEFORE any changes (for accurate diffing)
        self._snapshot_original_docs()

        # Check if this is a bootstrap run (no docs at all)
        existing_docs = self._get_doc_files()
        is_bootstrap = len(existing_docs) == 0

        if is_bootstrap:
            log.info("="*40)
            log.info("BOOTSTRAP MODE — No existing docs found!")
            log.info("Will create documentation from code analysis.")
            log.info("="*40)
            bootstrap_changes = self._bootstrap_docs_from_code()
            if bootstrap_changes:
                # After bootstrap, run normal improvement iterations on the new docs
                log.info("Bootstrap created %d docs — running improvement iterations...",
                         len(bootstrap_changes))
            else:
                log.warning("Bootstrap produced no docs — check code index.")
                return {
                    "mode": "bootstrap",
                    "iterations": [],
                    "total_docs_analyzed": 0,
                    "total_changes": 0,
                    "significant_changes": 0,
                    "pr_created": False,
                }
        else:
            bootstrap_changes = []

        # Load telemetry reference (may have been created during bootstrap)
        self._load_telemetry_reference()

        all_changes: List[Dict[str, str]] = list(bootstrap_changes)
        iteration_summaries = []

        for iteration in range(1, self.max_iterations + 1):
            log.info("--- Iteration %d/%d ---", iteration, self.max_iterations)

            changes_this_round = self._run_single_iteration(iteration)

            iteration_summaries.append({
                "iteration": iteration,
                "changes": len(changes_this_round),
            })

            if not changes_this_round:
                log.info("No changes in iteration %d — converged.", iteration)
                break

            # Merge changes (later iterations overwrite earlier ones for same file)
            existing_paths = {c["file_path"] for c in all_changes}
            for change in changes_this_round:
                if change["file_path"] in existing_paths:
                    # Replace the earlier version
                    all_changes = [c for c in all_changes if c["file_path"] != change["file_path"]]
                all_changes.append(change)

            log.info("Iteration %d: %d doc(s) improved", iteration, len(changes_this_round))

        # Filter by minimum diff threshold
        significant_changes = self._filter_by_diff(all_changes)

        result = {
            "mode": "bootstrap" if is_bootstrap else "incremental",
            "iterations": iteration_summaries,
            "total_docs_analyzed": len(self._get_doc_files()),
            "total_changes": len(all_changes),
            "significant_changes": len(significant_changes),
            "pr_created": False,
        }

        if not significant_changes:
            log.info("No significant changes found — skipping PR creation.")
            return result

        # Create the PR
        try:
            pr_result = self._create_improvement_pr(significant_changes)
            result["pr_created"] = True
            result["pr_url"] = pr_result.get("web_url", "")
            result["branch"] = pr_result.get("branch_name", "")
            result["files_changed"] = len(significant_changes)
            log.info("PR created: %s", pr_result.get("web_url", ""))
        except Exception as e:
            log.error("Failed to create PR: %s", e, exc_info=True)
            result["pr_error"] = str(e)

        log.info("DOC IMPROVEMENT CYCLE COMPLETE: %s", result)
        return result

    # ── Iteration logic ──────────────────────────────────────────────

    def _run_single_iteration(self, iteration: int) -> List[Dict[str, str]]:
        """Run one pass: analyze + improve existing docs, then discover new."""
        changes = []

        # Phase 1: Improve existing docs
        doc_files = self._get_doc_files()
        if not doc_files:
            log.info("No docs to improve in iteration %d (may be pre-bootstrap).", iteration)
        for doc_path in doc_files:
            fname = doc_path.name
            if fname in self.protected_docs:
                log.info("Skipping protected doc: %s", fname)
                continue

            try:
                change = self._improve_existing_doc(doc_path)
                if change:
                    changes.append(change)
            except Exception as e:
                log.error("Failed to improve %s: %s", fname, e, exc_info=True)

        # Phase 2: Discover undocumented features (only on first iteration)
        if iteration == 1:
            try:
                new_docs = self._discover_and_create_new_docs(doc_files)
                changes.extend(new_docs)
            except Exception as e:
                log.error("Feature discovery failed: %s", e, exc_info=True)

        return changes

    def _improve_existing_doc(self, doc_path: Path) -> Optional[Dict[str, str]]:
        """Analyze one doc against code and produce an improved version if needed."""
        fname = doc_path.name
        log.info("Analyzing doc: %s", fname)

        # Read current doc content
        doc_content = doc_path.read_text(encoding="utf-8", errors="replace")

        # Extract feature keywords from the doc for targeted code search
        feature_keywords = self._extract_feature_keywords(doc_content, fname)

        # Search code index for relevant source code
        code_snippets = self._search_code_for_feature(feature_keywords)
        if not code_snippets:
            log.info("No code found for %s — skipping", fname)
            return None

        # Also search the specific code folders from config
        folder_code = self._read_code_from_folders(feature_keywords)
        all_code = code_snippets + folder_code

        # Truncate to fit context window
        code_context = self._build_code_context(all_code, max_chars=60000)

        # Step 1: Analyze — ask code-reader to compare code vs doc
        analysis = self._analyze_doc_vs_code(doc_content, code_context, fname)
        if not analysis.get("has_changes"):
            log.info("Doc %s is up-to-date — no changes needed", fname)
            return None

        log.info("Doc %s needs improvement: %s", fname, analysis.get("summary", ""))

        # Step 2: Improve — ask writer LLM to produce improved doc
        improved_content = self._generate_improved_doc(
            doc_content, analysis, code_context, fname
        )
        if not improved_content or improved_content.strip() == doc_content.strip():
            log.info("Generated content identical to original for %s — skipping", fname)
            return None

        # Map to repo path
        repo_path = self._to_repo_path(doc_path)

        # Update local copy immediately (so next iteration sees improvements)
        doc_path.write_text(improved_content, encoding="utf-8")
        log.info("Updated local copy of %s", fname)

        return {
            "file_path": str(doc_path),
            "repo_path": repo_path,
            "new_content": improved_content,
            "summary": analysis.get("summary", f"Improved {fname}"),
        }

    def _discover_and_create_new_docs(self, existing_docs: List[Path]) -> List[Dict[str, str]]:
        """Discover features in code that lack documentation and create new docs."""
        new_docs = []

        existing_names = [d.name for d in existing_docs]
        log.info("Discovering undocumented features (existing docs: %d)", len(existing_names))

        # Get a broad sample of code to look for undocumented controllers/features
        broad_code = self.code_indexer.search("controller API endpoint handler", top_k=20)
        broad_code += self.code_indexer.search("DppWeb operation trigger validate", top_k=15)

        code_text = "\n\n---\n\n".join(
            f"// File: {c['source']}\n{c['text']}" for c in broad_code
        )[:40000]

        prompt = (
            f"## Existing documented features:\n"
            f"{chr(10).join(f'- {n}' for n in existing_names)}\n\n"
            f"## Source code samples:\n```\n{code_text}\n```\n\n"
            f"Identify features in the code that have NO corresponding documentation file."
        )

        raw = self.code_reader.generate(prompt, system=DISCOVERY_SYSTEM_PROMPT)

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                discovery = json.loads(json_match.group())
            else:
                log.info("No undocumented features discovered.")
                return []
        except Exception as e:
            log.warning("Failed to parse discovery result: %s", e)
            return []

        features = discovery.get("undocumented_features", [])
        log.info("Discovered %d potentially undocumented features", len(features))

        cap = self.max_new_docs_per_cycle
        for feature in features[:cap]:  # Cap at max_new_docs_per_cycle
            try:
                new_doc = self._create_new_feature_doc(feature)
                if new_doc:
                    new_docs.append(new_doc)
            except Exception as e:
                log.error("Failed to create doc for %s: %s",
                          feature.get("feature_name", "?"), e, exc_info=True)

        return new_docs

    def _create_new_feature_doc(self, feature: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Create a new documentation file for a discovered undocumented feature."""
        feature_name = feature["feature_name"]
        filename = feature.get("suggested_filename", f"Feature_{feature_name.replace(' ', '_')}.md")
        folder = feature.get("folder", "DPP")
        search_queries = feature.get("search_queries", [feature_name])

        log.info("Creating new doc: %s/%s", folder, filename)

        # Search code for this feature
        all_code = []
        for query in search_queries:
            hits = self.code_indexer.search(query, top_k=10)
            all_code.extend(hits)

        if not all_code:
            log.info("No code found for feature '%s' — skipping", feature_name)
            return None

        code_context = self._build_code_context(all_code, max_chars=50000)
        telemetry_ref = self._telemetry_ref or ""

        prompt = (
            f"## Feature to document: {feature_name}\n\n"
            f"## Code evidence:\n{feature.get('evidence', '')}\n\n"
            f"## Source code:\n```\n{code_context}\n```\n\n"
            f"## Telemetry reference (for KQL patterns):\n{telemetry_ref[:8000]}\n\n"
            f"## Standard template:\n{DOC_TEMPLATE_DESCRIPTION}\n\n"
            f"Create a comprehensive feature doc for '{feature_name}'."
        )

        content = self.writer_llm.generate(prompt, system=NEW_DOC_SYSTEM_PROMPT)
        content = self._strip_code_fences(content)

        if not content or len(content) < 200:
            log.info("Generated content too short for %s — skipping", feature_name)
            return None

        # Determine local path
        target_dir = self.docs_dir / "docs" / "agentKT" / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        doc_path = target_dir / filename

        # Write locally
        doc_path.write_text(content, encoding="utf-8")
        log.info("Created new doc: %s (%d chars)", doc_path, len(content))

        # Map to repo path
        repo_path = self._to_repo_path(doc_path)

        return {
            "file_path": str(doc_path),
            "repo_path": repo_path,
            "new_content": content,
            "summary": f"New doc: {feature_name}",
        }

    # ── Bootstrap (zero-docs) ─────────────────────────────────────────

    def _snapshot_original_docs(self):
        """Record the content of every existing doc BEFORE we start editing.
        This lets _filter_by_diff compare against the true original even after
        we write updated files locally during the cycle."""
        self._original_content.clear()
        for doc_path in self._get_doc_files():
            try:
                self._original_content[str(doc_path)] = doc_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                self._original_content[str(doc_path)] = None

    def _bootstrap_docs_from_code(self) -> List[Dict[str, str]]:
        """
        Bootstrap documentation from scratch when the repo has ZERO docs.

        Steps:
          1. Scan the entire code index to build a feature map
          2. Create the Architecture reference doc
          3. Create the Telemetry reference doc
          4. Create individual feature docs for every discovered feature

        Returns list of change dicts.
        """
        changes: List[Dict[str, str]] = []

        # ── Step 1: Broad code scan to map the codebase ─────────────
        log.info("[Bootstrap] Scanning codebase for feature map...")
        codebase_map = self._build_codebase_map()
        if not codebase_map:
            log.warning("[Bootstrap] Could not build codebase map — aborting.")
            return []

        features = codebase_map.get("features", [])
        arch_summary = codebase_map.get("architecture_summary", "")
        telemetry_patterns = codebase_map.get("telemetry_patterns", [])

        log.info("[Bootstrap] Found %d features to document.", len(features))

        # ── Step 2: Create Architecture doc ──────────────────────────
        log.info("[Bootstrap] Creating Architecture reference doc...")
        arch_change = self._bootstrap_architecture_doc(arch_summary, features)
        if arch_change:
            changes.append(arch_change)

        # ── Step 3: Create Telemetry doc ─────────────────────────────
        log.info("[Bootstrap] Creating Telemetry reference doc...")
        telemetry_change = self._bootstrap_telemetry_doc(telemetry_patterns)
        if telemetry_change:
            changes.append(telemetry_change)
            # Reload the telemetry ref so feature docs can reference it
            self._load_telemetry_reference()

        # ── Step 4: Create feature docs ──────────────────────────────
        # Sort by priority: high → medium → low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        features.sort(key=lambda f: priority_order.get(f.get("priority", "low"), 2))

        cap = self.max_new_docs_per_cycle
        created = 0
        for feature in features:
            if created >= cap:
                log.info("[Bootstrap] Hit max_new_docs_per_cycle (%d) — remaining "
                         "features will be picked up in the next cycle.", cap)
                break
            try:
                doc_change = self._create_new_feature_doc(feature)
                if doc_change:
                    changes.append(doc_change)
                    created += 1
            except Exception as e:
                log.error("[Bootstrap] Failed to create doc for %s: %s",
                          feature.get("feature_name", "?"), e, exc_info=True)

        log.info("[Bootstrap] Created %d docs (%d feature + %d reference).",
                 len(changes), created,
                 len(changes) - created)
        return changes

    def _build_codebase_map(self) -> Dict[str, Any]:
        """Scan the code index broadly and ask the LLM to map the codebase."""
        # Cast a wide net: search for many different code patterns
        search_queries = [
            "controller API endpoint route",
            "DppWeb operation trigger validate",
            "BackupInstance BackupPolicy ProtectedItem",
            "restore recovery point cross region",
            "job operation status tracking",
            "resource guard vault config",
            "protection container fabric engine",
            "data move resource move migration",
            "telemetry OpStats logging trace",
            "suspend resume stop delete protection",
            "tiering cost operation result",
            "workload item protection intent",
        ]

        all_hits = []
        seen_sources = set()
        for q in search_queries:
            hits = self.code_indexer.search(q, top_k=8)
            for h in hits:
                if h["source"] not in seen_sources:
                    seen_sources.add(h["source"])
                    all_hits.append(h)

        # Also pull from code folders on disk
        folder_hits = self._scan_code_folders_broadly()
        for h in folder_hits:
            if h["source"] not in seen_sources:
                seen_sources.add(h["source"])
                all_hits.append(h)

        if not all_hits:
            log.warning("[Bootstrap] No code found in index or folders.")
            return {}

        log.info("[Bootstrap] Collected %d unique code chunks for mapping.", len(all_hits))

        code_context = self._build_code_context(all_hits, max_chars=80000)

        prompt = (
            f"## Source code from the codebase (no docs exist yet):\n"
            f"```\n{code_context}\n```\n\n"
            f"Map the entire codebase into features that need documentation."
        )

        raw = self.code_reader.generate(prompt, system=BOOTSTRAP_CODEBASE_MAP_PROMPT)

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            log.error("[Bootstrap] Failed to parse codebase map: %s", e)

        return {}

    def _scan_code_folders_broadly(self) -> List[Dict[str, Any]]:
        """Scan configured code_folders and collect file summaries (first 2000 chars)."""
        results = []
        for folder in self.code_folders:
            folder_path = self.repo_clone_dir / folder
            if not folder_path.exists():
                continue
            for ext in [".cs", ".py"]:
                for code_file in folder_path.rglob(f"*{ext}"):
                    try:
                        text = code_file.read_text(encoding="utf-8", errors="replace")
                        # Only include files with substance (controllers, managers, etc.)
                        fname_lower = code_file.stem.lower()
                        interesting = any(kw in fname_lower for kw in [
                            "controller", "handler", "manager", "validator",
                            "impl", "service", "operation", "telemetry",
                        ])
                        if interesting or len(text) > 5000:
                            rel_path = str(code_file.relative_to(self.repo_clone_dir))
                            results.append({
                                "text": text[:2000],
                                "source": rel_path,
                                "score": 0.8,
                            })
                    except Exception:
                        pass
                    if len(results) >= 50:
                        return results
        return results

    def _bootstrap_architecture_doc(
        self, arch_summary: str, features: List[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """Create the top-level Architecture reference document."""
        # Gather a broad code sample for context
        code_hits = self.code_indexer.search("architecture namespace service layer", top_k=15)
        code_context = self._build_code_context(code_hits, max_chars=30000)

        feature_table = "\n".join(
            f"| {f['feature_name']} | {f.get('folder', '?')} | {f.get('suggested_filename', '?')} |"
            for f in features
        )

        prompt = (
            f"## Architecture summary (from code analysis):\n{arch_summary}\n\n"
            f"## Feature map:\n| Feature | Folder | Filename |\n|---------|--------|----------|\n{feature_table}\n\n"
            f"## Source code samples:\n```\n{code_context}\n```\n\n"
            f"Create the architecture reference document."
        )

        content = self.writer_llm.generate(prompt, system=BOOTSTRAP_ARCHITECTURE_PROMPT)
        content = self._strip_code_fences(content)

        if not content or len(content) < 300:
            log.warning("[Bootstrap] Architecture doc too short — skipping.")
            return None

        # Write to the root of the agentKT docs directory
        target_dir = self.docs_dir / "docs" / "agentKT"
        target_dir.mkdir(parents=True, exist_ok=True)
        doc_path = target_dir / "BackupMgmt_Architecture_Memory.md"
        doc_path.write_text(content, encoding="utf-8")

        log.info("[Bootstrap] Created architecture doc: %s (%d chars)", doc_path, len(content))

        repo_path = self._to_repo_path(doc_path)
        return {
            "file_path": str(doc_path),
            "repo_path": repo_path,
            "new_content": content,
            "summary": "New: Architecture & Memory reference document",
        }

    def _bootstrap_telemetry_doc(
        self, telemetry_patterns: List[str]
    ) -> Optional[Dict[str, str]]:
        """Create the Telemetry & Logging reference document."""
        # Search code for telemetry / logging patterns
        tel_hits = self.code_indexer.search("telemetry OpStats logging trace ActivityStats", top_k=15)
        tel_hits += self.code_indexer.search("Kusto table log message error code", top_k=10)
        code_context = self._build_code_context(tel_hits, max_chars=40000)

        patterns_text = "\n".join(f"- {p}" for p in telemetry_patterns) if telemetry_patterns else "(none found)"

        prompt = (
            f"## Telemetry patterns identified in code:\n{patterns_text}\n\n"
            f"## Source code with telemetry/logging:\n```\n{code_context}\n```\n\n"
            f"Create the Telemetry & Logging reference document."
        )

        content = self.writer_llm.generate(prompt, system=BOOTSTRAP_TELEMETRY_PROMPT)
        content = self._strip_code_fences(content)

        if not content or len(content) < 300:
            log.warning("[Bootstrap] Telemetry doc too short — skipping.")
            return None

        target_dir = self.docs_dir / "docs" / "agentKT"
        target_dir.mkdir(parents=True, exist_ok=True)
        doc_path = target_dir / "Telemetry_And_Logging_Reference.md"
        doc_path.write_text(content, encoding="utf-8")

        log.info("[Bootstrap] Created telemetry doc: %s (%d chars)", doc_path, len(content))

        repo_path = self._to_repo_path(doc_path)
        return {
            "file_path": str(doc_path),
            "repo_path": repo_path,
            "new_content": content,
            "summary": "New: Telemetry & Logging reference document",
        }

    # ── LLM interaction helpers ──────────────────────────────────────

    def _extract_feature_keywords(self, doc_content: str, filename: str) -> List[str]:
        """Extract search keywords from a doc for targeted code search."""
        keywords = []

        # From filename: Feature_BackupInstance.md → ["BackupInstance", "Backup Instance"]
        name_part = filename.replace("Feature_", "").replace(".md", "")
        keywords.append(name_part)
        # CamelCase split
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", name_part)
        if split != name_part:
            keywords.append(split)

        # Extract FM constants from doc (DppWebUpsertBackupInstance, etc.)
        fm_constants = re.findall(r"DppWeb\w+|Dpp\w+Controller|BMS\w+Controller", doc_content)
        keywords.extend(fm_constants[:10])

        # Extract class names mentioned in the doc
        class_names = re.findall(r"`(\w+(?:Controller|Impl|Manager|Handler|Validator))`", doc_content)
        keywords.extend(class_names[:10])

        # Extract route patterns
        routes = re.findall(r"/(\w+Instances?|backupPolicies|protectedItems|backupJobs)\b", doc_content)
        keywords.extend(routes[:5])

        return list(set(keywords))

    def _search_code_for_feature(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search the code vector store for relevant code snippets."""
        all_hits = []
        seen_sources = set()

        for kw in keywords[:8]:  # Limit to avoid too many searches
            hits = self.code_indexer.search(kw, top_k=5)
            for h in hits:
                source = h["source"]
                if source not in seen_sources:
                    seen_sources.add(source)
                    all_hits.append(h)

        return all_hits

    def _read_code_from_folders(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Read code directly from configured code_folders matching keywords."""
        results = []
        if not self.code_folders:
            return results

        for folder in self.code_folders:
            folder_path = self.repo_clone_dir / folder
            if not folder_path.exists():
                continue

            # Search for files containing any of the keywords
            for ext in [".cs", ".py"]:
                for code_file in folder_path.rglob(f"*{ext}"):
                    try:
                        text = code_file.read_text(encoding="utf-8", errors="replace")
                        # Check if file is relevant to any keyword
                        fname_lower = code_file.stem.lower()
                        if any(kw.lower() in fname_lower or kw.lower() in text[:2000].lower()
                               for kw in keywords[:5]):
                            rel_path = str(code_file.relative_to(self.repo_clone_dir))
                            # Take first 3000 chars as a snippet
                            results.append({
                                "text": text[:3000],
                                "source": rel_path,
                                "score": 0.9,
                            })
                    except Exception:
                        pass

                    if len(results) >= 15:
                        return results

        return results

    def _analyze_doc_vs_code(self, doc_content: str, code_context: str, filename: str) -> Dict[str, Any]:
        """Use code-reader LLM to analyze discrepancies between doc and code."""
        telemetry_ref = (self._telemetry_ref or "")[:6000]

        prompt = (
            f"## Document being analyzed: {filename}\n\n"
            f"## Current documentation:\n```markdown\n{doc_content[:12000]}\n```\n\n"
            f"## Source code (from the actual codebase):\n```\n{code_context[:40000]}\n```\n\n"
            f"## Telemetry reference (KQL patterns & table schemas):\n```\n{telemetry_ref}\n```\n\n"
            f"## Standard template sections expected:\n{DOC_TEMPLATE_DESCRIPTION}\n\n"
            f"Analyze the source code against the documentation. Identify ALL gaps, "
            f"inaccuracies, and missing debugging patterns."
        )

        raw = self.code_reader.generate(prompt, system=ANALYSIS_SYSTEM_PROMPT)

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            log.warning("Failed to parse analysis for %s: %s", filename, e)

        # If we can't parse JSON, check if the raw response mentions changes
        has_changes = any(word in raw.lower() for word in ["missing", "incorrect", "add", "update", "gap"])
        return {
            "has_changes": has_changes,
            "missing_items": [raw[:500]] if has_changes else [],
            "incorrect_items": [],
            "missing_kql_patterns": [],
            "summary": f"Analysis for {filename}",
            "severity": "medium" if has_changes else "low",
        }

    def _generate_improved_doc(
        self, doc_content: str, analysis: Dict, code_context: str, filename: str
    ) -> str:
        """Use the writer LLM to produce an improved doc based on the analysis."""
        telemetry_ref = (self._telemetry_ref or "")[:6000]

        analysis_text = json.dumps(analysis, indent=2, default=str)

        prompt = (
            f"## File: {filename}\n\n"
            f"## Analysis of needed improvements:\n```json\n{analysis_text}\n```\n\n"
            f"## Current document:\n{doc_content[:12000]}\n\n"
            f"## Source code evidence:\n```\n{code_context[:30000]}\n```\n\n"
            f"## Telemetry reference (for KQL queries):\n{telemetry_ref}\n\n"
            f"## Standard template:\n{DOC_TEMPLATE_DESCRIPTION}\n\n"
            f"Produce the complete improved document."
        )

        result = self.writer_llm.generate(prompt, system=IMPROVE_DOC_SYSTEM_PROMPT)
        return self._strip_code_fences(result)

    # ── Helpers ──────────────────────────────────────────────────────

    def _load_telemetry_reference(self):
        """Load the Telemetry_And_Logging_Reference.md once for all iterations."""
        telemetry_path = self.docs_dir / "docs" / "agentKT" / "Telemetry_And_Logging_Reference.md"
        if not telemetry_path.exists():
            # Try the flat path
            telemetry_path = self.docs_dir / "Telemetry_And_Logging_Reference.md"

        # Search recursively as fallback
        if not telemetry_path.exists():
            for f in self.docs_dir.rglob("Telemetry*Logging*.md"):
                telemetry_path = f
                break

        if telemetry_path.exists():
            self._telemetry_ref = telemetry_path.read_text(encoding="utf-8", errors="replace")
            log.info("Loaded telemetry reference: %s (%d chars)",
                     telemetry_path, len(self._telemetry_ref))
        else:
            log.warning("Telemetry reference not found — KQL patterns will be limited")
            self._telemetry_ref = ""

    def _get_doc_files(self) -> List[Path]:
        """Get all .md files from the docs directory (including DPP/ and RSV/ subfolders)."""
        if not self.docs_dir.exists():
            log.warning("Docs directory not found: %s", self.docs_dir)
            return []
        return sorted(self.docs_dir.rglob("*.md"))

    def _build_code_context(self, code_hits: List[Dict], max_chars: int = 60000) -> str:
        """Build a code context string from search hits, truncated to max_chars."""
        parts = []
        total = 0
        for hit in code_hits:
            snippet = f"// File: {hit['source']}\n{hit['text']}\n\n---\n\n"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "".join(parts)

    def _filter_by_diff(self, changes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter out changes that are below the minimum diff threshold."""
        significant = []
        for change in changes:
            file_path = change["file_path"]
            new_content = change["new_content"]

            # Read the original (from repo cache or local docs)
            original = self._read_original(file_path)
            if original is None:
                # New file — always significant
                significant.append(change)
                continue

            # Count diff lines
            diff_lines = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                lineterm="",
            ))
            added_lines = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            removed_lines = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
            net_change = added_lines + removed_lines

            if net_change >= self.min_diff_lines:
                significant.append(change)
                log.info("Significant change: %s (+%d/-%d)", Path(file_path).name, added_lines, removed_lines)
            else:
                log.info("Below threshold: %s (+%d/-%d < %d)", Path(file_path).name,
                         added_lines, removed_lines, self.min_diff_lines)

        return significant

    def _read_original(self, file_path: str) -> Optional[str]:
        """Read the original file content for diffing.

        Uses the pre-cycle snapshot if available so that files we created or
        updated during this cycle are correctly detected as new/changed.
        """
        # First check the pre-cycle snapshot (most accurate)
        if file_path in self._original_content:
            return self._original_content[file_path]  # may be content str or None

        # Not in snapshot → was not present when cycle started → it's a new file
        # But double-check the repo cache in case it exists upstream
        p = Path(file_path)
        for sp in self.cfg.get("azure_devops", {}).get("sync_paths", []):
            candidate = self.repo_clone_dir / sp / p.name
            if candidate.exists():
                return candidate.read_text(encoding="utf-8", errors="replace")

        return None

    def _to_repo_path(self, doc_path: Path) -> str:
        """Map a local doc path to the repo path for the PR."""
        # Try to construct path relative to the docs dir structure
        try:
            rel = doc_path.relative_to(self.docs_dir)
            # The repo stores these under the sync_paths (e.g., docs/agentKT/)
            sync_paths = self.cfg.get("azure_devops", {}).get("sync_paths", ["docs/agentKT"])
            base = sync_paths[0] if sync_paths else "docs/agentKT"
            return f"{base}/{rel}".replace("\\", "/")
        except ValueError:
            return str(doc_path).replace("\\", "/")

    def _create_improvement_pr(self, changes: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a single PR with all doc improvements."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"{self.branch_prefix}/{timestamp}"

        # Build file_changes for the batch PR
        file_changes = []
        for c in changes:
            # Detect whether this is a new file or an update to existing
            is_new = self._read_original(c["file_path"]) is None
            file_changes.append({
                "file_path": c["repo_path"],
                "new_content": c["new_content"],
                "summary": c["summary"],
                "changeType": "add" if is_new else "edit",
            })

        # Generate overall summary
        new_count = sum(1 for fc in file_changes if fc.get("changeType") == "add")
        edit_count = sum(1 for fc in file_changes if fc.get("changeType") == "edit")
        summaries = [c["summary"] for c in changes]

        if new_count == len(file_changes):
            # All new — this is a bootstrap run
            overall = f"Bootstrap documentation: {new_count} new docs created from code analysis"
        elif len(summaries) == 1:
            overall = summaries[0]
        else:
            parts = []
            if new_count:
                parts.append(f"{new_count} new")
            if edit_count:
                parts.append(f"{edit_count} updated")
            overall = f"Doc improvements: {', '.join(parts)}"

        # Use the existing batch PR helper
        return self.pr_helper.create_batch_correction_pr(
            file_changes=file_changes,
            overall_summary=overall,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from LLM output."""
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text
