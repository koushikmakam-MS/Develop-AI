"""
Debug Agent - connects to Azure Data Explorer (Kusto) via the Kusto MCP
server (or direct SDK fallback) and helps debug errors by reading debugging
procedures from docs and generating/running KQL queries.
"""

import logging
import re
from typing import Dict, List, Optional

from brain_ai.config import get_config
from brain_ai.kusto.client import KustoMCPClient
from brain_ai.llm_client import LLMClient
from brain_ai.vectorstore.indexer import DocIndexer

log = logging.getLogger(__name__)

# Pattern to detect GUIDs (TaskId, RequestId, etc.)
GUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Patterns that indicate the user already specified a time range
TIME_RANGE_PATTERNS = re.compile(
    r"(?i)("
    # "last 30 days", "last 7d", "last 24 hours"
    r"last\s+\d+\s*(h|hr|hour|d|day|m|min|minute|w|week|mo|month)s?"
    # "ago(30d)" KQL style
    r"|ago\(\d+[hdms]\)"
    # "30 days ago", "7 hours ago"
    r"|\d+\s*(hours?|days?|weeks?|months?)\s*ago"
    # "past 30 days"
    r"|past\s+\d+\s*(h|d|w|m|hours?|days?|weeks?|months?)"
    # "since yesterday"
    r"|since\s+(yesterday|last\s+week|last\s+month)"
    # standalone time references
    r"|today|yesterday|this\s+week"
    # "30 days", "7d", "24h", "90d" — standalone time durations
    r"|\b\d+\s*(h|hr|hours?|d|days?|w|weeks?|m|min|minutes?|mo|months?)\b"
    # "within 30 days", "in the last 30 days", "for 30 days"
    r"|(within|in\s+the\s+last|for\s+the\s+last|for)\s+\d+\s*(h|d|w|m|hours?|days?|weeks?|months?)"
    # "time range 30d", "timerange: 30 days"
    r"|time\s*range\s*:?\s*\d+\s*(h|d|w|m|hours?|days?|weeks?|months?)"
    r")"
)

# ---------------------------------------------------------------------------
# Phase 1: Standard Debugging Flow (always runs first)
# ---------------------------------------------------------------------------
STANDARD_FLOW_PROMPT = """You are the Debug Agent for the Azure Backup Management project.
Your ONLY job in this phase is to run the STANDARD 3-STEP DEBUGGING FLOW.
Do NOT use any documentation context yet — focus purely on the standard Kusto tables.

═══════════════════════════════════════════════════════════════
STANDARD DEBUGGING FLOW (follow for EVERY investigation)
═══════════════════════════════════════════════════════════════

**Step 1 — Find operations in `OperationStatsLocalAllClusters`**
Query `OperationStatsLocalAllClusters` for the time window and scope (subscription, vault, operation name,
TaskId, RequestId, etc.). Do NOT filter by `Result` — fetch ALL matching operations so you can
see both successes and failures.

```kql
OperationStatsLocalAllClusters
| where TIMESTAMP > ago(<timerange>)
| where <scope_filter>           // e.g. TaskId, RequestId, SubscriptionId, OperationName
| order by TIMESTAMP desc
```

If ALL results show `Result == true` → report: "All operations succeeded in this window." and STOP.
If ANY results show `Result == false` → note their `RequestId`/`TaskId` and proceed to Step 2.

**Step 2 — Get error traces from `TraceLogMessageAllClusters`** (only if Step 1 found failures)
Take the `RequestId`/`TaskId` from Step 1 and query `TraceLogMessageAllClusters`.
Filter for `Level` 1 (Critical) and 2 (Error) to isolate the actual failure messages.

```kql
TraceLogMessageAllClusters
| where TIMESTAMP > ago(<timerange>)
| where TaskId == "<taskId_from_step1>"
| where Level in (1, 2)
```

**Step 3 — Determine the failure service and root cause**
From the trace results:
- `Role` column → which service role logged the error (BMSWebRole, BMSDppTeeWorkerRole, etc.)
- `FileNameLineNumber` column → exact source file and line
- `Message` column → error description, exception details, inner exceptions
- `ErrorCode` → map using: UserError* = client 4xx, SystemError* = server 5xx,
  *Timeout* = dependency health, *NotFound* = missing resource, *NotAllowed* = state conflict

Level reference: 1=Critical, 2=Error, 3=Information (context), 4=Warning (retries/degraded)

═══════════════════════════════════════════════════════════════
GENERAL RULES
═══════════════════════════════════════════════════════════════

- Always explain what each query does before running it.
- When you want to EXECUTE a query, output it in this exact format:
  [EXECUTE_KQL]
  <your query here>
  [/EXECUTE_KQL]
- Only execute queries that are read-only (no .set, .drop, .alter, etc.)
- Always include a `where TIMESTAMP > ago(...)` filter in every KQL query.
- Prefer narrower ranges when possible for faster results.

CORRELATION KEYS:
- `TaskId` and `RequestId` are the primary keys for linking across all tables.
- Use them to join across tables.
- Once a TaskId/RequestId is known, EVERY subsequent query MUST include a
  `where TaskId == "<id>"` or `where RequestId == "<id>"` filter.
  Never run a broad unscoped query when you already have the correlation key.
"""

# ---------------------------------------------------------------------------
# Phase 2: Feature-Specific Deep Dive (runs after standard flow)
# ---------------------------------------------------------------------------
FEATURE_ANALYSIS_PROMPT = """You are the Debug Agent for the Azure Backup Management project.
You have already completed the STANDARD debugging flow (Steps 1-3) and have the results below.

Now use the **feature-specific documentation** provided to run deeper, more targeted queries
that give additional context about the failure. Match the failed operation to the right feature area:

- Failed "ConfigureProtection"/"EnableProtection" → BackupInstance feature queries
- Failed "Backup"/"AdhocBackup" → BackupInstance or Jobs feature queries
- Failed "Restore"/"CrossRegionRestore" → Restore / CRR feature queries
- Failed "CreatePolicy"/"UpdatePolicy" → BackupPolicy feature queries
- Failed "GetJob"/"ListJobs"/"TriggerJob" → Jobs feature queries
- Failed "Delete"/"Suspend"/"Resume" → SuspendResumeStopDelete feature queries
- Failed "ResourceGuard" related → ResourceGuards feature queries
- Failed "Vault" CRUD → VaultResource feature queries

IMPORTANT:
- The documentation context below contains KQL queries tailored for each feature area.
- Adapt those queries to the specific TaskId/RequestId/SubscriptionId from the standard flow results.
- Replace all `ago(<timerange>)` placeholders with the time range provided.
- Focus on queries that add NEW information beyond what the standard flow already found.

RULES:
- When you want to EXECUTE a query, output it in this exact format:
  [EXECUTE_KQL]
  <your query here>
  [/EXECUTE_KQL]
- Every KQL query MUST be scoped to the TaskId/RequestId from Phase 1 results.
  Never run a broad unscoped query — always include `where TaskId == "<id>"` or
  `where RequestId == "<id>"`.
- Summarize findings clearly with actionable next steps.
- If the docs don't have relevant feature-specific queries for this failure, say so and
  provide your best analysis based on the standard flow results alone.
- Only execute queries that are read-only.
"""

# Legacy alias for any external references
SYSTEM_PROMPT = STANDARD_FLOW_PROMPT


class DebugAgent:
    """Kusto-connected debug agent with RAG for debugging procedures."""

    # Regex for template placeholders the LLM may copy verbatim from prompts/docs.
    # Matches patterns like <subscriptionId>, <TaskId>, <taskId_from_step1>,
    # "<id>", "<TaskId>", <scope_filter>, etc.
    _PLACEHOLDER_RE = re.compile(
        r'["\']?<'                       # optional opening quote + <
        r'(?:subscriptionId|subscription_id|subId|sub_id'
        r'|taskId|task_id|taskId_from_step1|taskid_from_step1'
        r'|requestId|request_id|requestid'
        r'|id|guid'
        r'|scope_filter'
        r')>\'?"?',                      # > + optional closing quote
        re.IGNORECASE,
    )

    # Separate pattern for subscription-specific placeholders
    _SUB_PLACEHOLDER_RE = re.compile(
        r'["\']?<'
        r'(?:subscriptionId|subscription_id|subId|sub_id)'
        r'>["\']?',
        re.IGNORECASE,
    )

    # Pattern for TaskId/RequestId/generic id placeholders
    _TASKID_PLACEHOLDER_RE = re.compile(
        r'["\']?<'
        r'(?:taskId|task_id|taskId_from_step1|taskid_from_step1'
        r'|requestId|request_id|requestid'
        r'|id|guid)'
        r'>["\']?',
        re.IGNORECASE,
    )

    # Pattern to extract subscription ID from user input (looks for keyword context)
    _SUB_CONTEXT_RE = re.compile(
        r'(?:subscription(?:\s*id)?|sub(?:\s*id)?)\s*[:=]?\s*'
        r'["\']?(' + GUID_PATTERN.pattern + r')["\']?',
        re.IGNORECASE,
    )

    # Pattern to extract SubscriptionId from KQL results (column-value pairs)
    _SUB_RESULT_RE = re.compile(
        r'(?:SubscriptionId|subscriptionId|subscription_id)\s*[:=|]\s*'
        r'["\']?(' + GUID_PATTERN.pattern + r')["\']?',
        re.IGNORECASE,
    )

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.indexer = DocIndexer(cfg)

        # Use MCP client (tries MCP server first, falls back to direct SDK)
        self.kusto = KustoMCPClient(cfg)
        self._kusto_db = cfg["kusto"]["database"]
        self._session_time_range: str | None = None   # set per debug() call
        self._session_task_ids: set[str] = set()       # TaskId/RequestId GUIDs for this session
        self._session_subscription_ids: set[str] = set()  # SubscriptionId GUIDs for this session

        log.info(
            "DebugAgent initialized (Kusto MCP: %s, database: %s)",
            self.kusto.mcp_url,
            self._kusto_db,
        )

    def _extract_and_run_kql(self, text: str) -> Optional[str]:
        """Extract [EXECUTE_KQL]...[/EXECUTE_KQL] blocks and run them."""
        pattern = r"\[EXECUTE_KQL\]\s*\n(.*?)\n\s*\[/EXECUTE_KQL\]"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return None

        results = []
        for query in matches:
            query = self._replace_placeholders_in_kql(query)
            query = self._enforce_time_range(query)
            query = self._enforce_task_id_scope(query)
            log.info("Executing KQL (enforced): %s", query[:200])
            result = self.kusto.execute_kql(query)
            formatted = result.get("formatted", str(result))
            # Collect any new TaskId/RequestId GUIDs from the results
            self._collect_ids_from_results(formatted)
            results.append(f"**Query:**\n```kql\n{query}\n```\n**Results:**\n```\n{formatted}\n```")

        return "\n\n".join(results)

    def _needs_time_range(self, issue: str, conversation_history: List[Dict] | None) -> bool:
        """
        Check whether the user's message needs a time-range clarification.

        Every new KQL debugging session should start with a fresh time range.
        We only skip the prompt if:
          1. The current message already contains a time range, OR
          2. The current message is a direct reply to our time-range question
             (e.g. "go ahead", "use default").
        We intentionally do NOT look at older conversation history so each new
        debug query gets its own time-range prompt.
        """
        # If user already specified a time range in this message, no need to ask
        if TIME_RANGE_PATTERNS.search(issue):
            return False

        # If this is a follow-up response to our time-range question
        # (the immediately-previous assistant message asked for it)
        if conversation_history and len(conversation_history) >= 1:
            last_msg = conversation_history[-1]
            if (last_msg.get("role") == "assistant"
                    and "what time range should i use" in last_msg.get("content", "").lower()):
                # User is answering our time-range question → don't re-ask
                return False

        # Direct "go ahead" / "default" replies (user responding to time-range question)
        go_ahead_phrases = ["go ahead", "default", "just go", "90 day", "90d", "don't know",
                            "not sure", "idk", "no idea", "any", "whatever"]
        if any(phrase in issue.lower() for phrase in go_ahead_phrases):
            return False

        # Ask for time range if this looks like a debug/query request
        has_guid = bool(GUID_PATTERN.search(issue))
        has_debug_keywords = any(
            kw in issue.lower()
            for kw in ["taskid", "task id", "requestid", "request id", "query", "kql",
                        "debug", "error", "failure", "failed", "investigate", "check"]
        )

        return has_guid or has_debug_keywords

    def _extract_time_range(self, issue: str, conversation_history: List[Dict] | None) -> str | None:
        """
        Extract a concrete KQL time-range expression from the user's message
        or recent conversation history (e.g. "last 30 days" → "ago(30d)").
        Returns None if no time range is found.
        """
        # Check the current message first, then recent history
        sources = [issue]
        if conversation_history:
            sources.extend(msg.get("content", "") for msg in conversation_history[-6:])

        for text in sources:
            # "ago(30d)" already in KQL format
            m = re.search(r"ago\(\d+[dhms]\)", text, re.IGNORECASE)
            if m:
                return m.group()

            # "last 30 days", "last 7d", "last 24 hours", "past 90 days"
            m = re.search(
                r"(?:last|past)\s+(\d+)\s*(h|hr|hours?|d|days?|w|weeks?|m|min|minutes?|mo|months?)",
                text, re.IGNORECASE,
            )
            if m:
                num = m.group(1)
                unit = m.group(2).lower()
                if unit.startswith("h"):
                    return f"ago({num}h)"
                elif unit.startswith("d"):
                    return f"ago({num}d)"
                elif unit.startswith("w"):
                    return f"ago({int(num) * 7}d)"
                elif unit.startswith("mo"):
                    return f"ago({int(num) * 30}d)"
                elif unit.startswith("m"):
                    return f"ago({num}m)"

            # "30 days", "90d", "24h" standalone
            m = re.search(
                r"\b(\d+)\s*(h|hr|hours?|d|days?|w|weeks?|m|min|minutes?)\b",
                text, re.IGNORECASE,
            )
            if m:
                num = m.group(1)
                unit = m.group(2).lower()
                if unit.startswith("h"):
                    return f"ago({num}h)"
                elif unit.startswith("d"):
                    return f"ago({num}d)"
                elif unit.startswith("w"):
                    return f"ago({int(num) * 7}d)"
                elif unit.startswith("m"):
                    return f"ago({num}m)"

            # "go ahead", "default" → 90d
            if any(phrase in text.lower() for phrase in ["go ahead", "default", "just go", "not sure", "idk"]):
                return "ago(90d)"

        return None

    # ── helpers ──────────────────────────────────────────────────────────

    # Regex for any ago(...) expression in KQL, e.g. ago(30d), ago(1h), ago(7d)
    _AGO_RE = re.compile(r"ago\s*\(\s*[^)]+\s*\)", re.IGNORECASE)

    def _enforce_time_range(self, query: str) -> str:
        """Replace every `ago(...)` expression in *query* with the session time range.

        This guarantees that no matter what the LLM generates, the user's
        requested time window is always used.
        """
        if not self._session_time_range:
            return query
        return self._AGO_RE.sub(self._session_time_range, query)

    # Patterns for classifying GUIDs by surrounding keyword context
    _TASK_CONTEXT_RE = re.compile(
        r'(?:task(?:\s*id)?|request(?:\s*id)?)\s*[:=]?\s*'
        r'["\']?(' + GUID_PATTERN.pattern + r')["\']?',
        re.IGNORECASE,
    )

    def _classify_guids(self, text: str) -> dict[str, set[str]]:
        """Classify all GUIDs in *text* by context into subscription / task / unknown.

        Returns a dict with keys:
          - 'subscription': GUIDs near subscription/sub keywords
          - 'task': GUIDs near TaskId/RequestId keywords
          - 'unknown': GUIDs that couldn't be classified
        """
        all_guids = {g.lower() for g in GUID_PATTERN.findall(text)}
        if not all_guids:
            return {'subscription': set(), 'task': set(), 'unknown': set()}

        classified_sub: set[str] = set()
        classified_task: set[str] = set()

        # Check for subscription-context GUIDs
        for m in self._SUB_CONTEXT_RE.finditer(text):
            classified_sub.add(m.group(1).lower())

        # Check for task/request-context GUIDs
        for m in self._TASK_CONTEXT_RE.finditer(text):
            classified_task.add(m.group(1).lower())

        unknown = all_guids - classified_sub - classified_task
        return {
            'subscription': classified_sub,
            'task': classified_task,
            'unknown': unknown,
        }

    def _replace_placeholders_in_kql(self, query: str) -> str:
        """Replace literal placeholder tokens (e.g. <subscriptionId>, <TaskId>)
        that the LLM copied from docs/prompts with actual session values."""

        # Replace subscription placeholders
        if self._session_subscription_ids:
            sub_id = next(iter(self._session_subscription_ids))
            query = self._SUB_PLACEHOLDER_RE.sub(f'"{sub_id}"', query)

        # Replace TaskId / RequestId / generic id placeholders
        if self._session_task_ids:
            task_id = next(iter(sorted(self._session_task_ids)))
            query = self._TASKID_PLACEHOLDER_RE.sub(f'"{task_id}"', query)

        # Catch any remaining generic placeholders with the broadest pattern
        # but only if we have *some* ID to substitute
        if self._session_task_ids or self._session_subscription_ids:
            fallback_id = next(iter(sorted(
                self._session_task_ids or self._session_subscription_ids
            )))
            query = self._PLACEHOLDER_RE.sub(f'"{fallback_id}"', query)

        return query

    def _extract_task_ids(self, text: str) -> set[str]:
        """Pull all GUIDs from *text* that appear near TaskId/RequestId keywords."""
        ids: set[str] = set()
        # Direct GUID extraction from the user message
        ids.update(GUID_PATTERN.findall(text))
        return {g.lower() for g in ids}

    def _collect_ids_from_results(self, kql_results: str) -> None:
        """Scan KQL result text for TaskId / RequestId / SubscriptionId GUIDs and add to session."""
        if not kql_results:
            return

        # Capture SubscriptionId values from result columns
        for m in self._SUB_RESULT_RE.finditer(kql_results):
            sub_id = m.group(1).lower()
            if sub_id not in self._session_subscription_ids:
                self._session_subscription_ids.add(sub_id)
                log.info("Session SubscriptionId captured from results: %s", sub_id)

        # Capture TaskId / RequestId GUIDs (all remaining GUIDs)
        new_ids = {g.lower() for g in GUID_PATTERN.findall(kql_results)}
        # Don't re-add subscription IDs as task IDs
        new_ids -= self._session_subscription_ids
        if new_ids - self._session_task_ids:
            added = new_ids - self._session_task_ids
            self._session_task_ids.update(new_ids)
            log.info("Session TaskIds updated (+%d): %s", len(added),
                     ", ".join(list(added)[:5]))

    # Tables where TaskId / RequestId scoping makes sense
    _SCOPED_TABLES = re.compile(
        r"\b(TraceLogMessageAllClusters|OperationStatsLocalAllClusters"
        r"|BMSContainerRegistrationEvents|BMSJobEvents"
        r"|InternalBMSAdHocJobEvents|BMSExceptionTable)"
        r"\b",
        re.IGNORECASE,
    )

    # Detects an existing TaskId or RequestId filter in a query
    _HAS_TASK_FILTER = re.compile(
        r"\b(TaskId|RequestId)\s*(==|in\s*\(|has|contains)",
        re.IGNORECASE,
    )

    def _enforce_task_id_scope(self, query: str) -> str:
        """Inject a TaskId filter into the query if one is missing.

        Only applies when:
          - We have session TaskIds collected from the user or earlier results.
          - The query targets a known scoped table.
          - The query does NOT already contain a TaskId/RequestId filter.
        """
        if not self._session_task_ids:
            return query
        # Only scope queries that hit known tables
        if not self._SCOPED_TABLES.search(query):
            return query
        # Don't inject if the LLM already included a TaskId/RequestId filter
        if self._HAS_TASK_FILTER.search(query):
            return query

        # Build the filter clause
        ids = sorted(self._session_task_ids)
        if len(ids) == 1:
            filter_clause = f'| where TaskId == "{ids[0]}" or RequestId == "{ids[0]}"'
        else:
            id_list = ", ".join(f'"{i}"' for i in ids)
            filter_clause = f"| where TaskId in ({id_list}) or RequestId in ({id_list})"

        # Insert the filter right after the first `| where TIMESTAMP` line
        # so it sits with the other filters.
        ts_pattern = re.compile(
            r"(\|\s*where\s+TIMESTAMP\s*>\s*ago\s*\([^)]+\))",
            re.IGNORECASE,
        )
        m = ts_pattern.search(query)
        if m:
            insert_pos = m.end()
            query = query[:insert_pos] + "\n" + filter_clause + query[insert_pos:]
        else:
            # No TIMESTAMP filter found — append at end of query
            query = query.rstrip() + "\n" + filter_clause

        log.info("Injected TaskId scope into KQL query")
        return query

    def _run_phase(self, user_message: str, system_prompt: str,
                   history: List[Dict] | None) -> tuple[str, str | None]:
        """Run one LLM phase: generate → execute KQL → return (llm_text, kql_results)."""
        llm_text = self.llm.generate(
            message=user_message,
            system=system_prompt,
            history=history,
        )

        kql_results = None
        try:
            kql_results = self._extract_and_run_kql(llm_text)
        except Exception as e:
            kql_results = f"(Kusto query failed: {e})"

        return llm_text, kql_results

    def _clean_execute_blocks(self, text: str) -> str:
        """Remove [EXECUTE_KQL] blocks for cleaner display."""
        return re.sub(
            r"\[EXECUTE_KQL\].*?\[/EXECUTE_KQL\]",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    # ── main entry point ────────────────────────────────────────────────

    def debug(self, issue: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Two-phase debugging:

        Phase 1 — Standard Debugging Flow (always runs first)
          • Query OperationStatsLocalAll to find operations.
          • If failures found, query TraceLogMessageAllClusters for error traces.
          • Determine root cause from Role/Message/ErrorCode.
          • NO documentation context is used — purely table-based.

        Phase 2 — Feature-Specific Deep Dive (runs after Phase 1)
          • Search KT docs for feature-specific debugging queries.
          • Use the results from Phase 1 to pick the right feature area.
          • Run targeted queries from the docs for deeper analysis.
          • Combine everything into a final actionable summary.
        """
        # ── Pre-flight: time range ──────────────────────────────────────
        if self._needs_time_range(issue, conversation_history):
            guids = GUID_PATTERN.findall(issue)
            guid_str = ", ".join(f"`{g}`" for g in guids[:3]) if guids else "your query"
            return (
                f"I found {guid_str} in your request. Before I run KQL queries, "
                f"I need to know the **time range** to search.\n\n"
                f"**What time range should I use?**\n"
                f"- e.g., `last 1h`, `last 24h`, `last 7d`, `last 30d`\n"
                f"- Or say **\"go ahead\"** and I'll use the default (last 90 days)."
            )

        time_range = self._extract_time_range(issue, conversation_history) or "ago(90d)"
        self._session_time_range = time_range     # used by _enforce_time_range()

        # ── Classify GUIDs by context ───────────────────────────────────
        classified = self._classify_guids(issue)
        self._session_subscription_ids = classified['subscription']
        self._session_task_ids = classified['task']
        unknown_guids = classified['unknown']

        # If there are unclassified GUIDs, ask the user what they mean
        if unknown_guids and not classified['task'] and not classified['subscription']:
            guid_list = ", ".join(f"`{g}`" for g in sorted(unknown_guids))
            return (
                f"I found the following GUID(s) in your request: {guid_list}\n\n"
                f"I couldn't determine from context whether these are "
                f"**TaskId**, **RequestId**, or **SubscriptionId** values.\n\n"
                f"Please clarify by rephrasing, e.g.:\n"
                f"- `TaskId: <your-guid>`\n"
                f"- `RequestId: <your-guid>`\n"
                f"- `SubscriptionId: <your-guid>`\n\n"
                f"Or tell me which type each GUID is so I can run the right queries."
            )

        # If some GUIDs are unclassified but we already have classified ones,
        # treat the unknowns as TaskIds (most common case)
        if unknown_guids:
            log.info("Treating %d unclassified GUIDs as TaskIds: %s",
                     len(unknown_guids), ", ".join(sorted(unknown_guids)))
            self._session_task_ids.update(unknown_guids)

        log.info("Using time range: %s (enforced on all KQL queries)", time_range)
        if self._session_task_ids:
            log.info("Session TaskIds seeded: %s", ", ".join(sorted(self._session_task_ids)))
        if self._session_subscription_ids:
            log.info("Session SubscriptionIds seeded: %s", ", ".join(sorted(self._session_subscription_ids)))

        # ================================================================
        # PHASE 1: Standard Debugging Flow
        # ================================================================
        log.info("Phase 1: Standard debugging flow")

        task_id_note = ""
        if self._session_task_ids:
            ids_str = ", ".join(f"`{i}`" for i in sorted(self._session_task_ids))
            task_id_note = (
                f"## Session TaskId(s): {ids_str}\n"
                f"IMPORTANT: Every KQL query MUST include a `where TaskId` or "
                f"`where RequestId` filter scoped to these IDs.\n\n"
            )

        phase1_message = (
            f"## Kusto cluster: {self.cfg['kusto']['cluster_url']}\n"
            f"## Database: {self._kusto_db}\n\n"
            f"## Time range to use: `{time_range}`\n"
            f"IMPORTANT: Replace ALL `ago(<timerange>)` placeholders with `{time_range}` "
            f"in every query you generate.\n\n"
            f"{task_id_note}"
            f"## Issue to debug:\n{issue}\n\n"
            f"Run the STANDARD DEBUGGING FLOW (Steps 1 → 2 → 3). "
            f"Use [EXECUTE_KQL]...[/EXECUTE_KQL] to run queries."
        )

        phase1_llm, phase1_kql = self._run_phase(
            phase1_message, STANDARD_FLOW_PROMPT, conversation_history,
        )

        # If there were KQL results, feed them back for Phase 1 interpretation
        phase1_summary = phase1_llm  # default if no queries were executed
        if phase1_kql is not None:
            interp_history = list(conversation_history or [])
            interp_history.append({"role": "user", "content": phase1_message})
            interp_history.append({"role": "assistant", "content": phase1_llm})

            phase1_summary = self.llm.generate(
                message=(
                    f"## KQL Query Results (Standard Flow):\n\n{phase1_kql}\n\n"
                    "Analyze these standard-flow results. Summarize:\n"
                    "1. Which operations succeeded vs failed.\n"
                    "2. The error codes, roles and messages for any failures.\n"
                    "3. Your initial root-cause assessment.\n"
                    "4. Which feature area (BackupInstance, Jobs, Restore, Policy, etc.) "
                    "the failure belongs to so we can do a deeper dive."
                ),
                system=STANDARD_FLOW_PROMPT,
                history=interp_history,
            )

        # Build readable Phase 1 output
        clean_phase1_llm = self._clean_execute_blocks(phase1_llm)
        phase1_output_parts = ["## 🔍 Phase 1 — Standard Debugging Flow\n"]
        phase1_output_parts.append(clean_phase1_llm)
        if phase1_kql is not None:
            phase1_output_parts.append(f"\n\n### Standard Flow Results\n\n{phase1_kql}")
            phase1_output_parts.append(f"\n\n### Analysis\n\n{phase1_summary}")
        phase1_output = "\n".join(phase1_output_parts)

        # Check if Phase 1 found ALL successes → no need for Phase 2
        all_success_indicators = [
            "all operations succeeded",
            "all results show result == true",
            "everything worked",
            "no failures",
            "all succeeded",
        ]
        if any(ind in phase1_summary.lower() for ind in all_success_indicators):
            log.info("Phase 1 found no failures — skipping Phase 2")
            return phase1_output

        # ================================================================
        # PHASE 2: Feature-Specific Deep Dive (using KT docs)
        # ================================================================
        log.info("Phase 2: Feature-specific deep dive from KT docs")

        search_terms = f"debug troubleshoot error {issue}"
        hits = self.indexer.search(search_terms, top_k=6)

        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"--- Doc {i}: {hit['source']} (relevance: {hit['score']:.2f}) ---\n"
                f"{hit['text']}\n"
            )
        context_block = "\n".join(context_parts) if context_parts else "(No feature docs found)"

        # Refresh task_id_note — may have grown after Phase 1 results
        task_id_note2 = ""
        if self._session_task_ids:
            ids_str = ", ".join(f"`{i}`" for i in sorted(self._session_task_ids))
            task_id_note2 = (
                f"## Session TaskId(s): {ids_str}\n"
                f"IMPORTANT: Every KQL query MUST include a `where TaskId` or "
                f"`where RequestId` filter scoped to these IDs.\n\n"
            )

        phase2_message = (
            f"## Results from Standard Debugging Flow (Phase 1):\n\n{phase1_summary}\n\n"
            f"## Feature-specific documentation:\n\n{context_block}\n\n"
            f"## Kusto cluster: {self.cfg['kusto']['cluster_url']}\n"
            f"## Database: {self._kusto_db}\n\n"
            f"## Time range to use: `{time_range}`\n"
            f"IMPORTANT: Replace ALL `ago(<timerange>)` placeholders with `{time_range}`.\n\n"
            f"{task_id_note2}"
            f"## Original issue:\n{issue}\n\n"
            f"Based on the Phase 1 results, pick the relevant feature area from the docs and "
            f"run feature-specific queries to get deeper context. "
            f"Use [EXECUTE_KQL]...[/EXECUTE_KQL] to run queries.\n"
            f"After running queries, provide a FINAL consolidated summary with:\n"
            f"- Root cause\n"
            f"- Affected component / role\n"
            f"- Actionable next steps"
        )

        # Build Phase 2 history with Phase 1 context
        phase2_history = list(conversation_history or [])

        phase2_llm, phase2_kql = self._run_phase(
            phase2_message, FEATURE_ANALYSIS_PROMPT, phase2_history,
        )

        # If Phase 2 generated KQL, interpret the results
        phase2_summary = phase2_llm
        if phase2_kql is not None:
            interp_history2 = list(phase2_history)
            interp_history2.append({"role": "user", "content": phase2_message})
            interp_history2.append({"role": "assistant", "content": phase2_llm})

            phase2_summary = self.llm.generate(
                message=(
                    f"## Feature-Specific Query Results:\n\n{phase2_kql}\n\n"
                    f"## Phase 1 Summary (for context):\n\n{phase1_summary}\n\n"
                    "Provide a FINAL consolidated analysis combining the standard flow "
                    "and feature-specific results. Include:\n"
                    "1. **Root Cause** — what went wrong and why.\n"
                    "2. **Affected Component** — service role, code file, feature area.\n"
                    "3. **Error Classification** — UserError, SystemError, Timeout, etc.\n"
                    "4. **Actionable Next Steps** — what the team should do."
                ),
                system=FEATURE_ANALYSIS_PROMPT,
                history=interp_history2,
            )

        # Build readable Phase 2 output
        clean_phase2_llm = self._clean_execute_blocks(phase2_llm)
        phase2_output_parts = ["\n\n---\n\n## 📋 Phase 2 — Feature-Specific Deep Dive\n"]
        phase2_output_parts.append(clean_phase2_llm)
        if phase2_kql is not None:
            phase2_output_parts.append(f"\n\n### Feature-Specific Results\n\n{phase2_kql}")
            phase2_output_parts.append(f"\n\n### Final Analysis\n\n{phase2_summary}")
        elif phase2_llm != phase2_summary:
            phase2_output_parts.append(f"\n\n### Final Analysis\n\n{phase2_summary}")
        phase2_output = "\n".join(phase2_output_parts)

        return phase1_output + phase2_output
