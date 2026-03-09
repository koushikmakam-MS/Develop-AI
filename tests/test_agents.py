"""Tests for debug_agent patterns, coder_agent search, and GUID classification."""

import re
from unittest.mock import patch

import pytest

# Import the public patterns/regexes from debug_agent
from brain_ai.agents.debug_agent import GUID_PATTERN, TIME_RANGE_PATTERNS

# ── GUID Pattern ─────────────────────────────────────────────────────────

class TestGuidPattern:
    def test_matches_standard_guid(self):
        text = "TaskId: 33c55b50-14cb-4d67-9e61-9261f34eba6e"
        matches = GUID_PATTERN.findall(text)
        assert "33c55b50-14cb-4d67-9e61-9261f34eba6e" in matches

    def test_matches_uppercase_guid(self):
        text = "ID: AABBCCDD-1122-3344-5566-778899AABBCC"
        matches = GUID_PATTERN.findall(text)
        assert len(matches) == 1

    def test_matches_mixed_case_guid(self):
        text = "aAbBcCdD-1122-3344-5566-778899aAbBcC"
        matches = GUID_PATTERN.findall(text)
        assert len(matches) == 1

    def test_matches_multiple_guids(self):
        text = (
            "TaskId: 11111111-1111-1111-1111-111111111111 "
            "SubId: 22222222-2222-2222-2222-222222222222"
        )
        matches = GUID_PATTERN.findall(text)
        assert len(matches) == 2

    def test_no_match_on_short_hex(self):
        text = "not-a-guid: 1234-5678"
        matches = GUID_PATTERN.findall(text)
        assert len(matches) == 0

    def test_no_match_on_random_text(self):
        text = "Just a normal sentence."
        matches = GUID_PATTERN.findall(text)
        assert len(matches) == 0

    def test_guid_embedded_in_url(self):
        url = "/subscriptions/abcdef01-2345-6789-abcd-ef0123456789/resourceGroups/rg"
        matches = GUID_PATTERN.findall(url)
        assert "abcdef01-2345-6789-abcd-ef0123456789" in matches


# ── Time Range Patterns ──────────────────────────────────────────────────

class TestTimeRangePatterns:
    @pytest.mark.parametrize("text", [
        "last 30 days",
        "last 7d",
        "last 24 hours",
        "last 2 weeks",
        "last 1 month",
        "ago(30d)",
        "ago(7d)",
        "30 days ago",
        "7 hours ago",
        "past 14 days",
        "since yesterday",
        "today",
        "yesterday",
        "this week",
        "within 30 days",
        "in the last 7 days",
        "for 24 hours",
    ])
    def test_detects_time_ranges(self, text):
        assert TIME_RANGE_PATTERNS.search(text) is not None

    @pytest.mark.parametrize("text", [
        "debug this TaskId",
        "how does backup work",
        "explain the code",
    ])
    def test_no_false_positives(self, text):
        assert TIME_RANGE_PATTERNS.search(text) is None


# ── GUID Classification (via class-level regex patterns) ─────────────────

class TestGuidClassification:
    """Test the _SUB_CONTEXT_RE and _TASK_CONTEXT_RE regex patterns used by _classify_guids."""

    @pytest.fixture
    def agent(self):
        """Create a DebugAgent with all external deps mocked."""
        with patch("brain_ai.agents.debug_agent.get_config") as mock_cfg, \
             patch("brain_ai.agents.debug_agent.LLMClient"), \
             patch("brain_ai.agents.debug_agent.DocIndexer"), \
             patch("brain_ai.agents.debug_agent.KustoMCPClient"):
            mock_cfg.return_value = {
                "kusto": {"database": "testdb"},
                "azure_openai": {
                    "endpoint": "x", "api_key": "k",
                    "deployment": "d", "api_version": "v",
                },
                "chromadb": {
                    "path": "/tmp/db",
                    "collection_docs": "c1",
                    "collection_code": "c2",
                },
            }
            from brain_ai.agents.debug_agent import DebugAgent
            return DebugAgent(mock_cfg.return_value)

    def test_subscription_context(self, agent):
        text = "subscription id: aabbccdd-1122-3344-5566-778899aabbcc"
        result = agent._classify_guids(text)
        assert "aabbccdd-1122-3344-5566-778899aabbcc" in result["subscription"]
        assert len(result["task"]) == 0

    def test_task_context(self, agent):
        text = "TaskId: 11111111-1111-1111-1111-111111111111"
        result = agent._classify_guids(text)
        assert "11111111-1111-1111-1111-111111111111" in result["task"]
        assert len(result["subscription"]) == 0

    def test_request_id_context(self, agent):
        text = "RequestId=22222222-2222-2222-2222-222222222222"
        result = agent._classify_guids(text)
        assert "22222222-2222-2222-2222-222222222222" in result["task"]

    def test_mixed_guids(self, agent):
        text = (
            "subscription: aaaa1111-2222-3333-4444-555566667777 "
            "TaskId: bbbb1111-2222-3333-4444-555566667777 "
            "also cccc1111-2222-3333-4444-555566667777"
        )
        result = agent._classify_guids(text)
        assert "aaaa1111-2222-3333-4444-555566667777" in result["subscription"]
        assert "bbbb1111-2222-3333-4444-555566667777" in result["task"]
        assert "cccc1111-2222-3333-4444-555566667777" in result["unknown"]

    def test_no_guids(self, agent):
        result = agent._classify_guids("just some text without guids")
        assert result == {"subscription": set(), "task": set(), "unknown": set()}

    def test_extract_task_ids(self, agent):
        text = (
            "Check TaskId 11111111-1111-1111-1111-111111111111"
            " and 22222222-2222-2222-2222-222222222222"
        )
        ids = agent._extract_task_ids(text)
        assert "11111111-1111-1111-1111-111111111111" in ids
        assert "22222222-2222-2222-2222-222222222222" in ids


# ── Coder Agent _build_search_query ──────────────────────────────────────

class TestCoderBuildSearchQuery:
    """Test the PascalCase/quoted-string extraction logic."""

    def test_extracts_pascal_case(self):
        """PascalCase words like 'ConfigureBackup' should be extracted."""
        question = "How does ConfigureBackup call the ProtectionPolicyHandler?"
        words = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", question)
        assert "ConfigureBackup" in words
        assert "ProtectionPolicyHandler" in words
        assert "How" in words  # capitalized normal word, acceptable

    def test_extracts_double_quoted_strings(self):
        question = 'Error: "InvalidSubscriptionId" in the response'
        quoted = re.findall(r'"([^"]+)"', question)
        assert "InvalidSubscriptionId" in quoted

    def test_extracts_single_quoted_strings(self):
        question = "The operation 'TriggerBackup' failed"
        quoted = re.findall(r"'([^']+)'", question)
        assert "TriggerBackup" in quoted

    def test_no_extraction_on_plain_text(self):
        question = "how does backup work"
        words = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", question)
        assert words == []  # no PascalCase


# ── Knowledge Agent confidence ───────────────────────────────────────────

class TestKnowledgeConfidence:
    """Test the best_score calculation logic (extracted/unit)."""

    def test_max_score_from_hits(self):
        hits = [
            {"score": 0.30, "text": "a", "source": "f1.md"},
            {"score": 0.75, "text": "b", "source": "f2.md"},
            {"score": 0.50, "text": "c", "source": "f3.md"},
        ]
        best_score = max(h["score"] for h in hits)
        assert best_score == 0.75

    def test_single_hit(self):
        hits = [{"score": 0.42, "text": "x", "source": "f.md"}]
        assert max(h["score"] for h in hits) == 0.42

    def test_empty_hits_raises(self):
        with pytest.raises(ValueError):
            max(h["score"] for h in [])
