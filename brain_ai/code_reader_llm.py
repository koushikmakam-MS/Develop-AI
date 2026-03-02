"""
Code Reader LLM Client — uses gpt-5.1-codex-mini on Azure AI Foundry.

Separate from the main LLMClient so the doc-improver agent can use a
specialized code-reading model while the rest of the system keeps using
gpt-4o-mini.

NOTE: gpt-5.1-codex-mini uses the **Responses API** (not Chat Completions).
      We call `client.responses.create(...)` with `input` / `instructions`.
"""

import logging
import time

from openai import OpenAI, APIStatusError

from brain_ai.config import get_config

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 10


class CodeReaderLLM:
    """Azure OpenAI client configured from the `code_reader_llm` config section.

    Uses the Responses API for Codex models (gpt-5.1-codex-mini, etc.).
    Falls back to Chat Completions API when `use_responses_api` is False
    (e.g. when using a non-Codex model).
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()

        cr_cfg = cfg.get("code_reader_llm")
        if not cr_cfg:
            # Fall back to the main LLM section if no code_reader_llm configured
            log.warning("No code_reader_llm config found — falling back to main llm config.")
            cr_cfg = cfg["llm"]

        self.deployment = cr_cfg["model"]
        self.max_tokens = cr_cfg.get("max_tokens", 16384)

        # Auto-detect whether to use Responses API based on model name.
        # Codex models require Responses API; chat models use Chat Completions.
        self.use_responses_api = cr_cfg.get(
            "use_responses_api",
            "codex" in self.deployment.lower(),
        )

        endpoint = cr_cfg["endpoint"].rstrip("/")
        base_url = f"{endpoint}/openai/v1/"

        self.client = OpenAI(
            api_key=cr_cfg["api_key"],
            base_url=base_url,
        )

        api_mode = "Responses API" if self.use_responses_api else "Chat Completions"
        log.info(
            "CodeReaderLLM initialized — base_url=%s deployment=%s max_tokens=%d api=%s",
            base_url, self.deployment, self.max_tokens, api_mode,
        )

    def generate(self, message: str, system: str | None = None, history: list | None = None) -> str:
        """Generate a response using the code-reader model."""
        if self.use_responses_api:
            return self._generate_responses_api(message, system)
        return self._generate_chat_completions(message, system, history)

    # ── Responses API (for Codex models) ─────────────────────────────

    def _generate_responses_api(self, message: str, system: str | None = None) -> str:
        """Call the Responses API (required for gpt-5.1-codex-mini etc.)."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                kwargs: dict = {
                    "model": self.deployment,
                    "input": message,
                    "max_output_tokens": self.max_tokens,
                }
                if system:
                    kwargs["instructions"] = system

                resp = self.client.responses.create(**kwargs)
                return resp.output_text or ""
            except APIStatusError as e:
                if e.status_code == 429 and attempt < MAX_RETRIES:
                    wait = BASE_RETRY_DELAY * attempt
                    log.warning(
                        "Code-reader rate limited (attempt %d/%d). Retrying in %ds…",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    # ── Chat Completions API (fallback for non-Codex models) ────────

    def _generate_chat_completions(
        self, message: str, system: str | None = None, history: list | None = None,
    ) -> str:
        """Call the Chat Completions API (for gpt-4o-mini etc.)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history:
                role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
                messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content
            except APIStatusError as e:
                if e.status_code == 429 and attempt < MAX_RETRIES:
                    wait = BASE_RETRY_DELAY * attempt
                    log.warning(
                        "Code-reader rate limited (attempt %d/%d). Retrying in %ds…",
                        attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
