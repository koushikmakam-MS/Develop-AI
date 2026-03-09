"""
Knowledge Updater Agent - collects user corrections and updates documentation.

When the user corrects the model's understanding (e.g., "That's wrong, it actually
works like X..."), this agent:

1. Identifies which document(s) need updating based on the correction.
2. Uses the LLM to generate an updated version of the document.
3. Saves the correction locally and accumulates it in the session.
4. When the user says "agree" / "submit" / "create PR", batches ALL
   accumulated corrections into a single branch and Pull Request.

This ensures the knowledge base continuously improves from user feedback.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain_ai.config import get_config
from brain_ai.llm_client import LLMClient
from brain_ai.sync.devops_pr import AzureDevOpsPR
from brain_ai.vectorstore.indexer import DocIndexer

log = logging.getLogger(__name__)

# ── LLM Prompts ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Knowledge Updater Agent for the Azure Backup Management project.

Your job is to help update project documentation when a user provides a correction.

The user has told you that something in the project documentation is incorrect or incomplete.
You will be given:
1. The user's correction message
2. The conversation history (so you know what was previously discussed)
3. The relevant document content

Your task is to apply the correction to the document accurately.

Rules:
- Preserve the overall structure and formatting of the document.
- Only change the parts that need correcting based on the user's feedback.
- If adding new information, integrate it naturally into the existing sections.
- Keep the same Markdown formatting style.
- Do NOT remove existing correct information unless the user specifically says it's wrong.
- Be precise — apply exactly what the user said, don't over-edit.
"""

EXTRACTION_PROMPT = """You are extracting a documentation correction from a conversation.

IMPORTANT: A routing system has ALREADY determined that the user wants to correct or update
the Azure Backup Management documentation. Your job is ONLY to extract the details — do NOT
second-guess whether this is a correction. Set is_correction to true unless the message is
completely unrelated to documentation (e.g. "hello", "thanks", pure greetings).

Look at the conversation history for context. The correction may reference something said in
an earlier turn (e.g. "that's wrong" refers to the previous assistant answer).

Extract:
1. **correction**: What factual change is the user making? Include what was wrong and what the correct info is. If the earlier assistant answer contained wrong info, reference it.
2. **search_query**: Keywords to find the relevant doc (2-5 keywords, use Azure Backup domain terms).
3. **is_correction**: Almost always true. Only false for greetings or completely off-topic messages.

Respond in EXACTLY this JSON format (no other text):
{
  "is_correction": true,
  "correction": "The backup retry logic actually uses exponential backoff with a max of 3 retries, not 5.",
  "search_query": "backup retry logic exponential backoff"
}
"""

APPLY_CORRECTION_PROMPT = """You are updating a documentation file based on a user's correction.

## The Correction
{correction}

## Current Document Content
```markdown
{document_content}
```

## Instructions
Apply the correction to the document. Return ONLY the full updated document content in markdown.
Do not wrap it in code fences. Return the complete document — not just the changed section.
Preserve the original formatting, headings, and structure. Only change what needs correcting.
If the correction adds new information that doesn't fit in any existing section, add a new
appropriate section for it.
"""

SUMMARY_PROMPT = """Based on these corrections, generate a short (under 60 chars) summary
suitable for a git commit message and PR title. Be specific.

Corrections:
{corrections}

Return ONLY the summary text, nothing else."""

# Keywords that signal the user wants to submit the pending corrections
SUBMIT_KEYWORDS = [
    "agree", "submit", "create pr", "create the pr", "push it",
    "go ahead", "looks good", "lgtm", "ship it", "send it",
    "make the pr", "open pr", "open the pr", "submit pr",
    "yes create", "yes submit", "confirm", "finalize",
]

# Keywords that signal the user wants to discard pending corrections
DISCARD_KEYWORDS = [
    "discard", "drop", "cancel corrections", "never mind",
    "forget it", "clear corrections", "remove corrections",
]


class KnowledgeUpdaterAgent:
    """Collects user corrections during a session and submits them as one PR."""

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.indexer = DocIndexer(cfg)
        self.pr_helper = AzureDevOpsPR(cfg)

        # Path mappings
        self.local_docs_dir = Path(cfg["paths"]["docs_dir"]).resolve()
        self.repo_clone_dir = Path(cfg["paths"]["repo_clone_dir"]).resolve()

        ado = cfg["azure_devops"]
        self.sync_paths = ado.get("sync_paths", [])

        # ── Session state: accumulated corrections ──
        # Each entry: {doc_source, repo_path, correction, summary, new_content}
        self._pending_corrections: List[Dict[str, str]] = []

        log.info("KnowledgeUpdaterAgent ready.")

    # ── Public API ───────────────────────────────────────────

    def handle(self, message: str, conversation_history: List[Dict] | None = None) -> str:
        """
        Main entry point — called by BrainAgent.

        Two modes:
        1. **Correction mode**: user provides a correction -> extract, apply, stage locally.
        2. **Submit mode**: user says "agree" / "submit" -> create one PR for all staged corrections.
        """
        # Check if the user wants to discard pending corrections
        if self._is_discard_request(message):
            return self._discard_pending_corrections()

        # Check if the user wants to submit the pending corrections
        if self._is_submit_request(message):
            return self._submit_pending_corrections()

        # Otherwise, treat it as a new correction
        return self._process_correction(message, conversation_history)

    @property
    def pending_count(self) -> int:
        return len(self._pending_corrections)

    def pending_summary(self) -> str:
        """Human-readable summary of staged corrections."""
        if not self._pending_corrections:
            return "No pending corrections."
        lines = []
        for i, pc in enumerate(self._pending_corrections, 1):
            lines.append(f"{i}. **{pc['doc_source']}** — {pc['summary']}")
        return "\n".join(lines)

    def clear_pending(self):
        """Discard all staged corrections."""
        self._pending_corrections.clear()
        log.info("Cleared all pending corrections.")

    # ── Private: correction processing ───────────────────────

    def _process_correction(self, message: str, history: List[Dict] | None) -> str:
        """Extract, apply, and stage a single correction."""
        try:
            # Step 1: Extract the correction
            extraction = self._extract_correction(message, history)
            if not extraction.get("is_correction"):
                # If there are pending corrections, remind the user
                if self._pending_corrections:
                    return (
                        "I didn't detect a clear correction in your message.\n\n"
                        f"You have **{self.pending_count} pending correction(s)** staged:\n"
                        f"{self.pending_summary()}\n\n"
                        "Say **\"submit\"** to create a PR, or provide another correction."
                    )
                return (
                    "I didn't detect a clear correction in your message. "
                    "To update the documentation, please clearly state what's wrong "
                    "and what the correct information should be.\n\n"
                    "**Example:** _\"That's wrong — the backup retry actually uses "
                    "exponential backoff with max 3 retries, not 5.\"_"
                )

            correction = extraction["correction"]
            search_query = extraction["search_query"]

            # Step 2: Find relevant document
            hits = self.indexer.search(search_query, top_k=3)
            if not hits:
                return (
                    "I couldn't find a relevant document to update. "
                    "The documentation index might be empty. "
                    "Run `python run_sync.py && python run_index.py` first."
                )

            best_hit = hits[0]
            doc_source = best_hit["source"]
            log.info("Best matching document: %s (score: %.2f)", doc_source, best_hit["score"])

            # Step 3: Read the full document
            doc_content = self._read_full_document(doc_source)
            if not doc_content:
                return (
                    f"I found a matching document (`{doc_source}`) but couldn't read it. "
                    f"Make sure the docs are synced locally."
                )

            # Step 4: Apply the correction via LLM
            updated_content = self._apply_correction(correction, doc_content)
            if not updated_content or updated_content.strip() == doc_content.strip():
                return (
                    f"I analyzed `{doc_source}` but couldn't determine what to change. "
                    f"Could you be more specific about what needs correcting?"
                )

            # Step 5: Generate a short summary
            summary = self._generate_single_summary(correction)

            # Step 6: Map to repo path
            repo_path = self._to_repo_path(doc_source)
            if not repo_path:
                return (
                    f"I prepared the correction for `{doc_source}` but couldn't map it "
                    f"to the repo path. The file may need manual updating."
                )

            # Step 7: Update the local copy immediately
            self._update_local_copy(doc_source, updated_content)

            # Step 8: Stage the correction
            self._pending_corrections.append({
                "doc_source": doc_source,
                "repo_path": repo_path,
                "correction": correction,
                "summary": summary,
                "new_content": updated_content,
            })

            return (
                f"✏️ **Correction staged** (#{self.pending_count})\n\n"
                f"**File:** `{doc_source}`\n"
                f"**Change:** {summary}\n\n"
                f"I've updated my local knowledge immediately.\n\n"
                f"📋 **Pending corrections ({self.pending_count}):**\n"
                f"{self.pending_summary()}\n\n"
                f"You can:\n"
                f"- Provide more corrections\n"
                f"- Say **\"submit\"** or **\"agree\"** to create a PR with all corrections"
            )

        except Exception as e:
            log.error("Correction processing failed: %s", e, exc_info=True)
            return f"I understood your correction but hit an error: **{e}**"

    # ── Private: submit / PR creation ────────────────────────

    def _is_submit_request(self, message: str) -> bool:
        """Check if the user message is a request to submit pending corrections."""
        if not self._pending_corrections:
            return False
        msg_lower = message.strip().lower()
        return any(kw in msg_lower for kw in SUBMIT_KEYWORDS)

    def _is_discard_request(self, message: str) -> bool:
        """Check if the user wants to discard pending corrections."""
        if not self._pending_corrections:
            return False
        msg_lower = message.strip().lower()
        return any(kw in msg_lower for kw in DISCARD_KEYWORDS)

    def _discard_pending_corrections(self) -> str:
        """Drop all staged corrections."""
        count = self.pending_count
        self._pending_corrections.clear()
        return (
            f"🗑️ **Discarded {count} pending correction(s).**\n\n"
            f"Local doc copies were already updated with the corrections — "
            f"those will persist until the next `run_sync.py` re-pulls the originals.\n\n"
            f"No PR was created."
        )

    def _submit_pending_corrections(self) -> str:
        """Batch all staged corrections into one branch + one PR."""
        if not self._pending_corrections:
            return (
                "There are no pending corrections to submit. "
                "Provide a correction first, then say **\"submit\"**."
            )

        try:
            # Build file changes list for the batch PR
            file_changes = []
            for pc in self._pending_corrections:
                file_changes.append({
                    "file_path": pc["repo_path"],
                    "new_content": pc["new_content"],
                    "summary": pc["summary"],
                })

            # Generate an overall summary
            overall_summary = self._generate_overall_summary()

            # Create the batch PR
            pr_result = self.pr_helper.create_batch_correction_pr(
                file_changes=file_changes,
                overall_summary=overall_summary,
            )

            # Build the response
            file_list = "\n".join(
                f"  {i}. `{pc['doc_source']}` — {pc['summary']}"
                for i, pc in enumerate(self._pending_corrections, 1)
            )

            count = self.pending_count
            # Clear pending after successful submission
            self._pending_corrections.clear()

            return (
                f"✅ **Pull Request created with {count} correction(s)!**\n\n"
                f"**Files updated:**\n{file_list}\n\n"
                f"**Branch:** `{pr_result['branch_name']}`\n"
                f"**PR:** [{pr_result.get('title', 'View PR')}]({pr_result['web_url']})\n\n"
                f"A team member will review and merge it. "
                f"My local knowledge is already up to date."
            )

        except Exception as e:
            log.error("PR submission failed: %s", e, exc_info=True)
            return (
                f"Failed to create the Pull Request:\n\n"
                f"**Error:** {e}\n\n"
                f"Your {self.pending_count} correction(s) are still staged. "
                f"Fix the issue and say **\"submit\"** again.\n\n"
                f"_(Check the Azure DevOps PAT in `config.yaml` or repo permissions.)_"
            )

    # ── Private: LLM helpers ─────────────────────────────────

    def _extract_correction(self, message: str, history: List[Dict] | None) -> Dict[str, Any]:
        """Use LLM to extract correction details from the conversation."""
        history_text = ""
        if history:
            # Use last 10 messages for richer context (was 6, too tight)
            recent = history[-10:]
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:800]  # Allow more content per message
                history_text += f"**{role}:** {content}\n\n"

        user_msg = (
            f"## Conversation History\n{history_text}\n\n"
            f"## Latest User Message\n{message}\n\n"
            f"Remember: The routing system already classified this as a documentation "
            f"correction. Extract the correction details and set is_correction=true."
        )

        raw = self.llm.generate(
            message=user_msg,
            system=EXTRACTION_PROMPT,
            history=None,
        )

        log.debug("Extraction LLM raw response: %s", raw[:300])

        try:
            json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Trust the router — if extraction says false but we have
                # a non-empty correction text, override to true
                if not result.get("is_correction") and result.get("correction", "").strip():
                    log.info("Extraction said is_correction=false but has correction text; overriding to true.")
                    result["is_correction"] = True
                return result
        except Exception as e:
            log.warning("Failed to parse extraction: %s — raw: %s", e, raw[:200])

        # Fallback: if we can't parse the JSON but the router sent us here,
        # treat the raw message itself as the correction
        log.info("Extraction JSON parse failed; falling back to raw message as correction.")
        return {
            "is_correction": True,
            "correction": message,
            "search_query": " ".join(message.split()[:6]),
        }

    def _apply_correction(self, correction: str, document_content: str) -> str:
        """Use LLM to apply the correction to the document content."""
        prompt = APPLY_CORRECTION_PROMPT.format(
            correction=correction,
            document_content=document_content,
        )

        result = self.llm.generate(
            message=prompt,
            system=SYSTEM_PROMPT,
            history=None,
        )

        # Strip code fences the LLM might have added
        result = result.strip()
        if result.startswith("```markdown"):
            result = result[len("```markdown"):].strip()
        if result.startswith("```"):
            result = result[3:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        return result

    def _generate_single_summary(self, correction: str) -> str:
        """Generate a short summary for one correction."""
        prompt = (
            "Generate a short (under 50 chars) summary for this doc correction:\n\n"
            f"{correction}\n\nReturn ONLY the summary."
        )
        result = self.llm.generate(
            message=prompt,
            system="You generate concise git commit summaries.",
            history=None,
        )
        summary = result.strip().strip('"').strip("'")
        return summary[:50] if len(summary) > 50 else summary

    def _generate_overall_summary(self) -> str:
        """Generate an overall summary covering all pending corrections."""
        if len(self._pending_corrections) == 1:
            return self._pending_corrections[0]["summary"]

        corrections_text = "\n".join(
            f"- {pc['summary']}" for pc in self._pending_corrections
        )
        prompt = SUMMARY_PROMPT.format(corrections=corrections_text)
        result = self.llm.generate(
            message=prompt,
            system="You generate concise git commit summaries.",
            history=None,
        )
        summary = result.strip().strip('"').strip("'")
        return summary[:60] if len(summary) > 60 else summary

    # ── Private: file helpers ────────────────────────────────

    def _read_full_document(self, doc_source: str) -> Optional[str]:
        """Read the full content of a document given its source path from the indexer."""
        candidates = [
            self.local_docs_dir / doc_source,
            Path(doc_source),
            self.local_docs_dir / Path(doc_source).name,
        ]

        parts = Path(doc_source).parts
        for i in range(len(parts)):
            candidates.append(self.local_docs_dir / Path(*parts[i:]))

        for candidate in candidates:
            if candidate.is_file():
                log.debug("Reading document: %s", candidate)
                return candidate.read_text(encoding="utf-8", errors="replace")

        # Fallback: search by filename
        fname = Path(doc_source).name
        for f in self.local_docs_dir.rglob(fname):
            log.debug("Found document by name search: %s", f)
            return f.read_text(encoding="utf-8", errors="replace")

        log.warning("Could not find document: %s", doc_source)
        return None

    def _to_repo_path(self, doc_source: str) -> Optional[str]:
        """Map a local doc source path to the corresponding repo path."""
        doc_path = Path(doc_source)

        for sp in self.sync_paths:
            if doc_source.replace("\\", "/").startswith(sp):
                return doc_source.replace("\\", "/")

        fname = doc_path.name
        for sp in self.sync_paths:
            src = self.repo_clone_dir / sp
            if src.exists():
                for f in src.rglob(fname):
                    repo_path = str(f.relative_to(self.repo_clone_dir)).replace("\\", "/")
                    log.info("Mapped '%s' -> repo path '%s'", doc_source, repo_path)
                    return repo_path

        candidate = self.repo_clone_dir / doc_source
        if candidate.is_file():
            return doc_source.replace("\\", "/")

        log.warning("Could not map '%s' to a repo path.", doc_source)
        return None

    def _update_local_copy(self, doc_source: str, new_content: str):
        """Update the local copy of the document so the index stays current."""
        doc_path = Path(doc_source)

        candidates = [
            self.local_docs_dir / doc_source,
            Path(doc_source),
        ]
        parts = Path(doc_source).parts
        for i in range(len(parts)):
            candidates.append(self.local_docs_dir / Path(*parts[i:]))

        for candidate in candidates:
            if candidate.is_file():
                candidate.write_text(new_content, encoding="utf-8")
                log.info("Updated local copy: %s", candidate)
                return

        fname = doc_path.name
        for f in self.local_docs_dir.rglob(fname):
            f.write_text(new_content, encoding="utf-8")
            log.info("Updated local copy (by name): %s", f)
            return

        log.warning("Could not update local copy for: %s", doc_source)
