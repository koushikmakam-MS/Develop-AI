"""Tests for brain_ai.llm_client — message construction and retry logic."""

from unittest.mock import MagicMock, patch

import pytest

from brain_ai.llm_client import BASE_RETRY_DELAY, LLMClient


def _make_cfg(**overrides):
    """Return a minimal config dict for LLMClient."""
    cfg = {
        "llm": {
            "model": "test-model",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "max_tokens": 100,
        }
    }
    cfg["llm"].update(overrides)
    return cfg


class TestLLMClientInit:
    def test_init_sets_deployment_and_max_tokens(self):
        client = LLMClient(_make_cfg())
        assert client.deployment == "test-model"
        assert client.max_tokens == 100

    def test_default_max_tokens(self):
        cfg = _make_cfg()
        del cfg["llm"]["max_tokens"]
        client = LLMClient(cfg)
        assert client.max_tokens == 4096

    def test_base_url_constructed_correctly(self):
        client = LLMClient(_make_cfg())
        assert client.client.base_url is not None
        assert "openai/v1" in str(client.client.base_url)


class TestLLMClientGenerate:
    def setup_method(self):
        self.client = LLMClient(_make_cfg())

    def _mock_response(self, text="Hello"):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = text
        return resp

    def test_message_construction_no_system_no_history(self):
        self.client.client.chat.completions.create = MagicMock(
            return_value=self._mock_response()
        )
        self.client.generate("Hi there")
        call_kwargs = self.client.client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi there"

    def test_message_construction_with_system(self):
        self.client.client.chat.completions.create = MagicMock(
            return_value=self._mock_response()
        )
        self.client.generate("Hello", system="You are helpful")
        messages = self.client.client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_message_construction_with_history(self):
        self.client.client.chat.completions.create = MagicMock(
            return_value=self._mock_response()
        )
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "model", "content": "prev answer"},  # "model" should map to "assistant"
        ]
        self.client.generate("follow up", history=history)
        messages = self.client.client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "prev question"
        assert messages[1]["role"] == "assistant"  # "model" → "assistant"
        assert messages[2]["role"] == "user"

    def test_returns_response_text(self):
        self.client.client.chat.completions.create = MagicMock(
            return_value=self._mock_response("Test response")
        )
        result = self.client.generate("Hello")
        assert result == "Test response"

    @patch("brain_ai.llm_client.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        from openai import APIStatusError

        # First call → 429, second call → success
        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.headers = {}

        self.client.client.chat.completions.create = MagicMock(
            side_effect=[
                APIStatusError("rate limited", response=error_resp, body=None),
                self._mock_response("OK after retry"),
            ]
        )
        result = self.client.generate("retry test")
        assert result == "OK after retry"
        mock_sleep.assert_called_once_with(BASE_RETRY_DELAY * 1)

    def test_raises_non_429_immediately(self):
        from openai import APIStatusError

        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.headers = {}

        self.client.client.chat.completions.create = MagicMock(
            side_effect=APIStatusError("server error", response=error_resp, body=None)
        )
        with pytest.raises(APIStatusError):
            self.client.generate("should fail")
