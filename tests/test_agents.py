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


# ── New Document Creation ────────────────────────────────────────────────

class TestNewDocCreation:
    """Verify that the agent creates new docs when no matching doc exists."""

    @pytest.fixture
    def updater_cfg(self, tmp_path):
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
            "protected_docs": ["BackupMgmt_Architecture_Memory.md"],
        }

    @pytest.fixture
    def updater(self, updater_cfg):
        with patch("brain_ai.agents.knowledge_updater_agent.LLMClient"), \
             patch("brain_ai.agents.knowledge_updater_agent.DocIndexer"), \
             patch("brain_ai.agents.knowledge_updater_agent.AzureDevOpsPR"):
            from brain_ai.agents.knowledge_updater_agent import KnowledgeUpdaterAgent
            return KnowledgeUpdaterAgent(updater_cfg)

    def test_no_hits_offers_new_doc(self, updater):
        """When no search hits, the agent should offer to create a new doc."""
        updater.indexer.search = MagicMock(return_value=[])
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "The new retry feature uses circuit breaker pattern",
            "search_query": "retry circuit breaker",
        })

        result = updater.handle("The retry uses circuit breaker pattern", [])
        assert "create doc" in result.lower() or "new document" in result.lower()
        assert updater._pending_new_doc is not None
        assert updater._pending_new_doc["topic"] == "retry circuit breaker"
        # New format: corrections is a list
        assert updater._pending_new_doc["corrections"] == [
            "The new retry feature uses circuit breaker pattern"
        ]

    def test_accumulates_corrections_for_new_doc(self, updater):
        """Subsequent messages should accumulate into _pending_new_doc.corrections."""
        updater.indexer.search = MagicMock(return_value=[])

        # First message — creates the offer
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "Billing uses consumption-based model",
            "search_query": "billing management",
        })
        result1 = updater.handle("Billing uses consumption-based model", [])
        assert "create doc" in result1.lower()
        assert len(updater._pending_new_doc["corrections"]) == 1

        # Second message — accumulates
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "It also supports reserved instances and pre-pay",
            "search_query": "billing reserved instances",
        })
        result2 = updater.handle("It also supports reserved instances and pre-pay", [])
        assert "2 pieces" in result2.lower()
        assert len(updater._pending_new_doc["corrections"]) == 2
        assert "reserved instances" in updater._pending_new_doc["corrections"][1]

        # Third message — accumulates more
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "Cost alerts are sent via Azure Monitor",
            "search_query": "billing cost alerts",
        })
        result3 = updater.handle("Cost alerts are sent via Azure Monitor", [])
        assert "3 pieces" in result3.lower()
        assert len(updater._pending_new_doc["corrections"]) == 3

    def test_only_protected_hits_offers_new_doc(self, updater):
        """When all hits are protected, the agent should offer to create a new doc."""
        updater.indexer.search = MagicMock(return_value=[
            {"source": "BackupMgmt_Architecture_Memory.md",
             "score": 0.90, "text": "arch overview"},
        ])
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "The architecture now includes circuit breaker",
            "search_query": "architecture circuit breaker",
        })

        result = updater.handle("The arch has circuit breaker now", [])
        assert "create doc" in result.lower()
        assert "protected" in result.lower()
        assert updater._pending_new_doc is not None

    def test_is_create_doc_request_true(self, updater):
        updater._pending_new_doc = {"topic": "x", "corrections": ["y"], "search_query": "z"}
        assert updater._is_create_doc_request("create doc")
        assert updater._is_create_doc_request("Yes, create a doc please")
        assert updater._is_create_doc_request("new doc")

    def test_is_create_doc_request_false_without_pending(self, updater):
        updater._pending_new_doc = None
        assert not updater._is_create_doc_request("create doc")

    def test_create_doc_generates_file(self, updater, tmp_path):
        """Full flow: no hit → offer → user says 'create doc' → file created."""
        updater._pending_new_doc = {
            "topic": "retry circuit breaker",
            "corrections": ["The new retry feature uses circuit breaker pattern with 3 retries"],
            "search_query": "retry circuit breaker",
        }
        updater.llm.generate = MagicMock(side_effect=[
            "# Feature: Retry Circuit Breaker\n\n> **Purpose**: Circuit breaker for retries.\n",
            '{"filename": "Feature_RetryCircuitBreaker.md", "folder": "DPP"}',
        ])
        updater.indexer.index_file = MagicMock(return_value=3)

        result = updater.handle("create doc", [])
        assert "new document created" in result.lower()
        assert updater.pending_count == 1
        assert updater._pending_new_doc is None

        # New docs must be flagged as is_new so the PR uses changeType="add"
        assert updater._pending_corrections[0]["is_new"] is True

        # Verify file was written
        doc_path = Path(updater.local_docs_dir) / "DPP" / "Feature_RetryCircuitBreaker.md"
        assert doc_path.exists()
        content = doc_path.read_text()
        assert "Circuit Breaker" in content

    def test_create_doc_root_folder(self, updater, tmp_path):
        """When folder is empty, doc is placed in the root docs dir."""
        updater._pending_new_doc = {
            "topic": "cross cutting concern",
            "corrections": ["General info about error handling patterns"],
            "search_query": "error handling patterns",
        }
        updater.llm.generate = MagicMock(side_effect=[
            "# Feature: Error Handling Patterns\n\nGeneral error handling.\n",
            '{"filename": "Feature_ErrorHandling.md", "folder": ""}',
        ])
        updater.indexer.index_file = MagicMock(return_value=2)

        result = updater.handle("create doc", [])
        assert "new document created" in result.lower()

        doc_path = Path(updater.local_docs_dir) / "Feature_ErrorHandling.md"
        assert doc_path.exists()

    def test_create_doc_uses_all_accumulated_corrections(self, updater, tmp_path):
        """When multiple corrections are accumulated, all are sent to the LLM."""
        updater._pending_new_doc = {
            "topic": "billing management",
            "corrections": [
                "Billing uses consumption-based model",
                "It supports reserved instances and pre-pay",
                "Cost alerts are sent via Azure Monitor",
            ],
            "search_query": "billing management",
        }
        updater.llm.generate = MagicMock(side_effect=[
            "# Feature: Billing Management\n\nConsumption + reserved instances.\n",
            '{"filename": "Feature_BillingManagement.md", "folder": "DPP"}',
        ])
        updater.indexer.index_file = MagicMock(return_value=3)

        result = updater.handle("create doc", [])
        assert "new document created" in result.lower()

        # Verify that all 3 corrections were sent to the LLM in the prompt
        llm_call_args = updater.llm.generate.call_args_list[0]
        prompt_text = llm_call_args.kwargs.get("message", llm_call_args[0][0] if llm_call_args[0] else "")
        assert "consumption-based model" in prompt_text
        assert "reserved instances" in prompt_text
        assert "Cost alerts" in prompt_text

    def test_suggest_filename_fallback(self, updater):
        """If LLM returns garbage, fallback produces a valid filename."""
        updater.llm.generate = MagicMock(return_value="not valid json at all")
        filename, folder = updater._suggest_doc_filename("backup retry logic")
        assert filename.endswith(".md")
        assert "Backup" in filename or "backup" in filename.lower()
        assert folder in ("DPP", "RSV", "")

    def test_suggest_filename_accepts_any_folder(self, updater):
        """Any single-name folder the LLM suggests is accepted."""
        updater.llm.generate = MagicMock(
            return_value='{"filename": "Feature_X.md", "folder": "NewArea"}'
        )
        filename, folder = updater._suggest_doc_filename("some topic")
        assert filename == "Feature_X.md"
        assert folder == "NewArea"  # accepted, new folder is fine

    def test_suggest_filename_sanitises_folder(self, updater):
        """Folder with path traversal or special chars is sanitised."""
        updater.llm.generate = MagicMock(
            return_value='{"filename": "Feature_Y.md", "folder": "../../etc"}'
        )
        filename, folder = updater._suggest_doc_filename("harmless")
        assert ".." not in folder
        assert "/" not in folder
        assert "\\" not in folder

    def test_suggest_filename_strips_path_components(self, updater):
        """Filename with directory traversal components is sanitised."""
        updater.llm.generate = MagicMock(
            return_value='{"filename": "../../etc/passwd.md", "folder": ""}'
        )
        filename, folder = updater._suggest_doc_filename("harmless topic")
        # Path(name).name strips directory parts -> "passwd.md"
        assert ".." not in filename
        assert "/" not in filename
        assert "\\" not in filename
        assert filename.endswith(".md")

    def test_save_new_document_rejects_path_traversal(self, updater):
        """_save_new_document refuses to write outside docs_dir."""
        with pytest.raises(ValueError, match="Refusing to write outside docs_dir"):
            updater._save_new_document("evil.md", "../../..", "bad content")

    def test_save_new_document_writes_in_allowed_child(self, updater):
        """_save_new_document succeeds for a valid child folder."""
        doc_path, _ = updater._save_new_document(
            "Feature_Test.md", "DPP", "# Test Content\n"
        )
        written = Path(updater.local_docs_dir) / "DPP" / "Feature_Test.md"
        assert written.exists()
        assert written.read_text() == "# Test Content\n"

    def test_save_new_document_creates_new_folder(self, updater):
        """_save_new_document creates a brand-new subfolder if needed."""
        doc_src, _ = updater._save_new_document(
            "Feature_NewArea.md", "BrandNew", "# New Area\n"
        )
        written = Path(updater.local_docs_dir) / "BrandNew" / "Feature_NewArea.md"
        assert written.exists()
        assert written.read_text() == "# New Area\n"

    def test_pending_new_doc_cleared_after_create(self, updater):
        """After creating a new doc, _pending_new_doc should be None."""
        updater._pending_new_doc = {
            "topic": "test topic",
            "corrections": ["test content"],
            "search_query": "test",
        }
        updater.llm.generate = MagicMock(side_effect=[
            "# Test Doc\n\nContent.\n",
            '{"filename": "Feature_Test.md", "folder": ""}',
        ])
        updater.indexer.index_file = MagicMock(return_value=1)

        updater.handle("create doc", [])
        assert updater._pending_new_doc is None
        assert updater.pending_count == 1

    def test_submit_sets_changetype_add_for_new_docs(self, updater):
        """When submitting, new docs get changeType='add', edits get 'edit'."""
        # Stage a regular edit (no is_new)
        updater._pending_corrections.append({
            "doc_source": "DPP/Feature_Jobs.md",
            "repo_path": "docs/agentKT/DPP/Feature_Jobs.md",
            "correction": "Fixed a typo",
            "summary": "Typo fix",
            "new_content": "# Updated\n",
        })
        # Stage a new doc (is_new=True)
        updater._pending_corrections.append({
            "doc_source": "NewFolder/Feature_New.md",
            "repo_path": "docs/agentKT/NewFolder/Feature_New.md",
            "correction": "New document for: new topic",
            "summary": "New doc: Feature_New.md",
            "new_content": "# Brand New\n",
            "is_new": True,
        })

        # Capture what create_batch_correction_pr receives
        updater.pr_helper.create_batch_correction_pr = MagicMock(return_value={
            "branch_name": "test-branch",
            "web_url": "https://example.com/pr/1",
            "title": "Test PR",
        })
        updater._generate_overall_summary = MagicMock(return_value="mixed changes")

        updater._submit_pending_corrections()

        call_args = updater.pr_helper.create_batch_correction_pr.call_args
        file_changes = call_args.kwargs.get("file_changes") or call_args[1].get("file_changes")
        assert file_changes[0]["changeType"] == "edit"
        assert file_changes[1]["changeType"] == "add"

    def test_regular_correction_has_no_is_new(self, updater):
        """Regular corrections (existing doc edits) should not have is_new."""
        # Set up a doc that exists locally for the correction flow
        doc_dir = Path(updater.local_docs_dir) / "DPP"
        doc_dir.mkdir(parents=True, exist_ok=True)
        doc_file = doc_dir / "Feature_Jobs.md"
        doc_file.write_text("# Original Content\n", encoding="utf-8")

        updater.indexer.search = MagicMock(return_value=[
            {"source": "DPP/Feature_Jobs.md", "score": 0.90, "text": "jobs docs"},
        ])
        updater._extract_correction = MagicMock(return_value={
            "is_correction": True,
            "correction": "Jobs can now retry automatically",
            "search_query": "jobs retry",
        })
        updater._apply_correction = MagicMock(return_value="# Updated Jobs\n\nRetry info.\n")
        updater._generate_single_summary = MagicMock(return_value="Added retry info")
        updater._to_repo_path = MagicMock(return_value="docs/agentKT/DPP/Feature_Jobs.md")

        updater.handle("Jobs can now retry automatically", [])
        assert updater.pending_count == 1
        assert updater._pending_corrections[0].get("is_new") is None


class TestToRepoPath:
    """Verify _to_repo_path resolves files in repo clone AND local docs_dir."""

    @pytest.fixture
    def updater(self, tmp_path):
        docs = tmp_path / "docs" / "agentKT"
        docs.mkdir(parents=True)
        repo = tmp_path / ".repo"
        repo.mkdir()
        cfg = {
            "llm": {"model": "m", "endpoint": "e", "api_key": "k", "max_tokens": 4096},
            "vectorstore": {"persist_directory": ".chromadb", "collection_name": "c",
                            "embedding_model": "all-MiniLM-L6-v2", "chunk_size": 1000,
                            "chunk_overlap": 200},
            "paths": {"docs_dir": str(docs), "repo_clone_dir": str(repo)},
            "azure_devops": {"repo_url": "https://x", "branch": "main",
                             "sync_paths": ["docs/agentKT"], "pat": "fake"},
            "protected_docs": [],
        }
        with patch("brain_ai.agents.knowledge_updater_agent.LLMClient"), \
             patch("brain_ai.agents.knowledge_updater_agent.DocIndexer"), \
             patch("brain_ai.agents.knowledge_updater_agent.AzureDevOpsPR"):
            from brain_ai.agents.knowledge_updater_agent import KnowledgeUpdaterAgent
            return KnowledgeUpdaterAgent(cfg)

    def test_maps_file_in_repo_clone(self, updater):
        """File found in .repo_cache/docs/agentKT/ maps correctly."""
        repo_doc = updater.repo_clone_dir / "docs" / "agentKT" / "DPP" / "Feature_Jobs.md"
        repo_doc.parent.mkdir(parents=True, exist_ok=True)
        repo_doc.write_text("# Jobs", encoding="utf-8")

        result = updater._to_repo_path("Feature_Jobs.md")
        assert result == "docs/agentKT/DPP/Feature_Jobs.md"

    def test_maps_sync_path_prefix(self, updater):
        """doc_source that already starts with sync_path is returned as-is."""
        result = updater._to_repo_path("docs/agentKT/DPP/Feature_X.md")
        assert result == "docs/agentKT/DPP/Feature_X.md"

    def test_maps_new_doc_from_local_docs_dir(self, updater):
        """Newly created doc in local_docs_dir (not in repo clone) maps via sync_path."""
        local_file = Path(updater.local_docs_dir) / "NewFolder" / "Feature_New.md"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("# New", encoding="utf-8")

        result = updater._to_repo_path("NewFolder/Feature_New.md")
        assert result == "docs/agentKT/NewFolder/Feature_New.md"

    def test_maps_new_doc_root_from_local_docs_dir(self, updater):
        """New doc at the root of local_docs_dir maps correctly."""
        local_file = Path(updater.local_docs_dir) / "Feature_Root.md"
        local_file.write_text("# Root", encoding="utf-8")

        result = updater._to_repo_path("Feature_Root.md")
        assert result == "docs/agentKT/Feature_Root.md"

    def test_maps_by_filename_search_in_local_docs(self, updater):
        """Even with only the filename, finds the file via rglob in local docs."""
        local_file = Path(updater.local_docs_dir) / "Deep" / "Sub" / "Feature_Deep.md"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("# Deep", encoding="utf-8")

        result = updater._to_repo_path("Feature_Deep.md")
        assert result == "docs/agentKT/Deep/Sub/Feature_Deep.md"

    def test_returns_none_for_nonexistent(self, updater):
        """Completely unknown file returns None."""
        result = updater._to_repo_path("Feature_DoesNotExist.md")
        assert result is None
