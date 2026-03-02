from openai import OpenAI, APIStatusError
import logging
import time

from brain_ai.config import get_config

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 10

class LLMClient:
    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = get_config()

        llm_cfg = cfg["llm"]

        # IMPORTANT: for Azure OpenAI, this MUST be the DEPLOYMENT NAME
        # not the catalog model name.
        self.deployment = llm_cfg["model"]
        self.max_tokens = llm_cfg.get("max_tokens", 4096)

        endpoint = llm_cfg["endpoint"].rstrip("/")

        # Azure OpenAI v1 OpenAI-compatible base url
        base_url = f"{endpoint}/openai/v1/"

        self.client = OpenAI(
            api_key=llm_cfg["api_key"],
            base_url=base_url,
        )

        log.info(
            "LLMClient initialized (Azure OpenAI v1 via OpenAI SDK) base_url=%s deployment=%s",
            base_url, self.deployment
        )

    def generate(self, message: str, system=None, history=None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for msg in history:
                role = "assistant" if msg["role"] == "model" else msg["role"]
                messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.deployment,   # deployment name required on Azure
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content
            except APIStatusError as e:
                if e.status_code == 429 and attempt < MAX_RETRIES:
                    wait = BASE_RETRY_DELAY * attempt
                    log.warning("Rate limited (attempt %d/%d). Retrying in %ds...", attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    raise
