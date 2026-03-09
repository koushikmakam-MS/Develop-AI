"""Tests for debug_agent patterns, coder_agent search, GUID classification,
and protected-doc filtering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ── Coder Agent _build_search_queries (multi-query) ─────────────────────

class TestCoderBuildSearchQueries:
    """Test the multi-query extraction logic used by the refactored coder agent."""

    @pytest.fixture
    def agent(self):
        """Create a CoderAgent with mocked external deps."""
        with patch("brain_ai.agents.coder_agent.get_config") as mock_cfg, \
             patch("brain_ai.agents.coder_agent.LLMClient"), \
             patch("brain_ai.agents.coder_agent.CodeIndexer") as mock_idx:
            mock_idx_instance = MagicMock()
            mock_idx_instance.collection.count.return_value = 100
            mock_idx.return_value = mock_idx_instance
            mock_cfg.return_value = {
                "llm": {
                    "model": "m", "endpoint": "e",
                    "api_key": "k", "max_tokens": 4096,
                },
                "vectorstore": {"persist_directory": ".chromadb"},
                "code_index": {"collection_name": "c"},
                "paths": {"repo_clone_dir": ".repo"},
                "coder_agent": {
                    "top_k_per_query": 12,
                    "max_context_chars": 80000,
                    "min_relevance_score": 0.20,
                },
            }
            from brain_ai.agents.coder_agent import CoderAgent
            return CoderAgent(mock_cfg.return_value)

    def test_plain_question_produces_one_query(self, agent):
        queries = agent._build_search_queries("how does backup work")
        assert len(queries) >= 1
        assert queries[0] == "how does backup work"

    def test_pascal_case_adds_symbol_query(self, agent):
        queries = agent._build_search_queries(
            "How does ConfigureBackup call ProtectionPolicyHandler?"
        )
        # Should have at least the raw question + a PascalCase-only query
        assert len(queries) >= 2
        symbol_query = queries[1]
        assert "ConfigureBackup" in symbol_query
        assert "ProtectionPolicyHandler" in symbol_query

    def test_quoted_strings_become_a_query(self, agent):
        queries = agent._build_search_queries(
            'Error: "InvalidSubscriptionId" in the response'
        )
        assert any("InvalidSubscriptionId" in q for q in queries)

    def test_operation_words_generate_call_chain_query(self, agent):
        queries = agent._build_search_queries(
            "How does ConfigureBackup flow through the system?"
        )
        # Should produce a call-chain query with handler/orchestrator
        chain_queries = [q for q in queries if "handler" in q.lower() or "orchestrator" in q.lower()]
        assert len(chain_queries) >= 1

    def test_no_duplicate_queries(self, agent):
        queries = agent._build_search_queries("explain the code")
        assert len(queries) == len(set(queries))

    def test_multi_query_search_deduplicates(self, agent):
        """Hits from different queries with same source+text keep best score."""
        hit_a = {"text": "class Foo:" + " " * 80, "source": "a.cs",
                 "score": 0.5, "chunk_label": "", "file_ext": ".cs"}
        hit_b = {**hit_a, "score": 0.8}  # same chunk, higher score
        agent.code_indexer.search = MagicMock(side_effect=[[hit_a], [hit_b]])
        merged = agent._multi_query_search(["q1", "q2"])
        assert len(merged) == 1
        assert merged[0]["score"] == 0.8

    def test_multi_query_filters_low_score(self, agent):
        """Chunks below min_relevance_score are excluded."""
        low = {"text": "x" * 81, "source": "a.cs",
               "score": 0.05, "chunk_label": "", "file_ext": ".cs"}
        agent.code_indexer.search = MagicMock(return_value=[low])
        merged = agent._multi_query_search(["q1"])
        assert len(merged) == 0

    def test_context_budget_limits_output(self, agent):
        """_build_context_block should stop adding chunks when budget exhausted."""
        big_hits = [
            {"text": "x" * 20_000, "source": f"f{i}.cs",
             "score": 0.9 - i * 0.01, "chunk_label": "", "file_ext": ".cs"}
            for i in range(10)
        ]
        agent._max_context_chars = 50_000
        context, sources = agent._build_context_block(big_hits)
        assert len(context) <= 55_000  # small overhead for headers is OK
        assert len(sources) < 10  # not all 10 fit


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


# ── Protected Docs (un-editable files) ──────────────────────────────────

class TestProtectedDocs:
    """Verify that agents respect the protected_docs config."""

    @pytest.fixture
    def updater_cfg(self, tmp_path):
        """Config with protected_docs and mocked paths."""
        docs = tmp_path / "docs" / "agentKT"
        docs.mkdir(parents=True)
        return {
            "llm": {"model": "m", "endpoint": "e", "api_key": "k", "max_tokens": 4096},
            "vectorstore": {"persist_directory": ".chromadb", "collection_name": "c",
                            "embedding_model": "all-MiniLM-L6-v2", "chunk_size": 1000,
                            "chunk_overlap": 200},
            "paths": {"docs_dir": str(docs), "repo_clone_dir": str(tmp_path / ".repo")},
            "azure_devops": {"repo_url": "https://x", "branch": "main",
                             "sync_paths": ["docs/agentKT"], "pat": "fake"},
            "protected_docs": [
                "BackupMgmt_Architecture_Memory.md",
                "Telemetry_And_Logging_Reference.md",
            ],
        }

    @pytest.fixture
    def updater(self, updater_cfg):
        with patch("brain_ai.agents.knowledge_updater_agent.LLMClient"), \
             patch("brain_ai.agents.knowledge_updater_agent.DocIndexer"), \
             patch("brain_ai.agents.knowledge_updater_agent.AzureDevOpsPR"):
            from brain_ai.agents.knowledge_updater_agent import KnowledgeUpdaterAgent
            return KnowledgeUpdaterAgent(updater_cfg)

    def test_protected_docs_loaded_from_config(self, updater):
        assert "BackupMgmt_Architecture_Memory.md" in updater.protected_docs
        assert "Telemetry_And_Logging_Reference.md" in updater.protected_docs

    def test_is_protected_true_for_listed_file(self, updater):
        assert updater.is_protected("docs/agentKT/BackupMgmt_Architecture_Memory.md")

    def test_is_protected_false_for_feature_doc(self, updater):
        assert not updater.is_protected("docs/agentKT/DPP/Feature_BackupPolicy.md")

    def test_is_protected_matches_filename_only(self, updater):
        # Regardless of path prefix, filename match is enough
        assert updater.is_protected("some/deep/path/Telemetry_And_Logging_Reference.md")

    def test_default_protected_when_config_empty(self):
        """If config has no protected_docs, a safe default is used."""
        cfg = {
            "llm": {"model": "m", "endpoint": "e", "api_key": "k", "max_tokens": 4096},
            "vectorstore": {"persist_directory": ".chromadb", "collection_name": "c",
                            "embedding_model": "all-MiniLM-L6-v2", "chunk_size": 1000,
                            "chunk_overlap": 200},
            "paths": {"docs_dir": "docs", "repo_clone_dir": ".repo"},
            "azure_devops": {"repo_url": "https://x", "branch": "main",
                             "sync_paths": ["docs/agentKT"], "pat": "fake"},
            # No protected_docs key at all
        }
        with patch("brain_ai.agents.knowledge_updater_agent.LLMClient"), \
             patch("brain_ai.agents.knowledge_updater_agent.DocIndexer"), \
             patch("brain_ai.agents.knowledge_updater_agent.AzureDevOpsPR"):
            from brain_ai.agents.knowledge_updater_agent import KnowledgeUpdaterAgent
            agent = KnowledgeUpdaterAgent(cfg)
            assert "BackupMgmt_Architecture_Memory.md" in agent.protected_docs

    def test_correction_skips_protected_doc(self, updater):
        """When the best vector match is a protected doc, the agent should skip it
        and use the next-best editable hit, or return a helpful message."""
        # Simulate search returning only protected docs
        updater.indexer.search = MagicMock(return_value=[
            {"source": "docs/agentKT/BackupMgmt_Architecture_Memory.md",
             "score": 0.90, "text": "architecture overview"},
        ])
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "The retry count is 3 not 5",
            "search_query": "retry count",
        })

        result = updater.handle("The retry is 3 not 5", [])
        assert "protected" in result.lower()
        assert updater.pending_count == 0

    def test_correction_falls_through_to_editable_hit(self, updater, tmp_path):
        """When first hit is protected but second is editable, the second is used."""
        # Write a real doc so _read_full_document can find it
        docs_dir = Path(updater.local_docs_dir)
        dpp = docs_dir / "DPP"
        dpp.mkdir(parents=True, exist_ok=True)
        feature_doc = dpp / "Feature_BackupPolicy.md"
        feature_doc.write_text("# Backup Policy\nRetry count: 5\n")

        updater.indexer.search = MagicMock(return_value=[
            {"source": "BackupMgmt_Architecture_Memory.md",
             "score": 0.90, "text": "architecture overview"},
            {"source": "DPP/Feature_BackupPolicy.md",
             "score": 0.75, "text": "backup policy details"},
        ])
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "The retry count is 3 not 5",
            "search_query": "retry count backup policy",
        })
        updater._apply_correction = MagicMock(
            return_value="# Backup Policy\nRetry count: 3\n"
        )
        updater._generate_single_summary = MagicMock(return_value="Fix retry count")
        updater._to_repo_path = MagicMock(return_value="docs/agentKT/DPP/Feature_BackupPolicy.md")

        result = updater.handle("The retry is 3 not 5", [])
        assert "staged" in result.lower() or "correction" in result.lower()
        assert updater.pending_count == 1
        assert "Feature_BackupPolicy.md" in updater._pending_corrections[0]["doc_source"]
