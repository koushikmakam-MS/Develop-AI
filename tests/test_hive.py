"""Tests for the Agent Hive Architecture — Hive, HiveRegistry, HiveRouter.

Covers:
  - Hive creation & config merging
  - Scope matching & topic keywords
  - Delegation signal parsing
  - HiveRegistry construction & lookup
  - HiveRouter hive selection (single & multi)
  - Cross-hive delegation protocol
  - Synthesis of multi-hive responses
  - Backward compatibility (hives disabled)
  - Config isolation between hives
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Shared test config ──────────────────────────────────────────────────

def _global_cfg():
    """Minimal global config with hive definitions for testing."""
    return {
        "llm": {
            "model": "test",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "key",
        },
        "vectorstore": {
            "persist_directory": ".chromadb",
            "collection_name": "default_docs",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "paths": {
            "docs_dir": "docs/default",
            "repo_clone_dir": ".repo_cache",
        },
        "kusto": {
            "cluster_url": "https://cluster",
            "database": "db",
            "mcp_url": "http://127.0.0.1:8701",
        },
        "azure_devops": {
            "pat": "test-pat",
            "repo_url": "https://x.visualstudio.com/P/_git/R",
            "branch": "main",
        },
        "code_index": {
            "collection_name": "default_code",
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "sync_paths": ["src"],
        },
        "agents": {
            "enabled": [],
            "knowledge_confidence_threshold": 0.45,
            "doc_gap_threshold": 0.50,
        },
        "hives": {
            "enabled": True,
            "default_hive": "bms",
            "definitions": {
                "bms": {
                    "display_name": "Backup Management Service",
                    "description": "Primary BCDR hive for backup management.",
                    "scope": {
                        "topics": [
                            "backup management",
                            "DPP",
                            "RSV",
                            "backup policies",
                            "restore operations",
                            "backup job",
                            "backup failure",
                            "UserError",
                        ],
                        "kusto_tables": [
                            "OperationStatsLocalAllClusters",
                        ],
                    },
                    "paths": {
                        "docs_dir": "docs/agentKT",
                    },
                    "vectorstore": {
                        "collection_name": "bms_docs",
                    },
                    "code_index": {
                        "collection_name": "bms_code",
                        "sync_paths": ["src/BMS"],
                    },
                    "agents": {
                        "enabled": ["knowledge", "debug", "coder", "knowledge_updater"],
                    },
                },
                "common": {
                    "display_name": "Common Libraries",
                    "description": "Shared SDK utilities and data contracts.",
                    "scope": {
                        "topics": [
                            "common libraries",
                            "SDK utilities",
                            "shared data contracts",
                        ],
                        "kusto_tables": [],
                    },
                    "paths": {
                        "docs_dir": "docs/common",
                    },
                    "vectorstore": {
                        "collection_name": "common_docs",
                    },
                    "code_index": {
                        "collection_name": "common_code",
                        "sync_paths": ["src/Common"],
                    },
                    "agents": {
                        "enabled": ["knowledge", "coder"],
                    },
                },
                "protection": {
                    "display_name": "Protection & Workload Agents",
                    "description": "Workload plugins for VM, SQL, SAP.",
                    "scope": {
                        "topics": [
                            "protection agents",
                            "VM backup",
                            "SQL backup",
                        ],
                        "kusto_tables": ["OperationStatsLocalAllClusters"],
                    },
                    "paths": {
                        "docs_dir": "docs/protection",
                    },
                    "vectorstore": {
                        "collection_name": "prot_docs",
                    },
                    "code_index": {
                        "collection_name": "prot_code",
                    },
                    "agents": {
                        "enabled": ["knowledge", "debug", "coder"],
                    },
                },
            },
        },
    }


def _bms_only_cfg():
    """Config with only the BMS hive (simulates single-hive mode)."""
    cfg = _global_cfg()
    cfg["hives"]["definitions"] = {
        "bms": cfg["hives"]["definitions"]["bms"],
    }
    return cfg


# ══════════════════════════════════════════════════════════════════════
# Hive class tests
# ══════════════════════════════════════════════════════════════════════


class TestHive:
    """Tests for brain_ai.hive.hive.Hive."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_hive_creation(self, *_mocks):
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        hive = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)

        assert hive.name == "bms"
        assert hive.display_name == "Backup Management Service"
        assert "backup management" in hive.topics
        assert "DPP" in hive.topics
        assert len(hive.topics) == 8
        assert "OperationStatsLocalAllClusters" in hive.kusto_tables

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_config_merging_overrides_paths(self, *_mocks):
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        hive = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)

        # Hive-level docs_dir should override global
        assert hive._merged_cfg["paths"]["docs_dir"] == "docs/agentKT"
        # Global repo_clone_dir should be inherited
        assert hive._merged_cfg["paths"]["repo_clone_dir"] == ".repo_cache"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_config_merging_overrides_vectorstore(self, *_mocks):
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        hive = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)

        assert hive._merged_cfg["vectorstore"]["collection_name"] == "bms_docs"
        # Other vectorstore keys inherited
        assert hive._merged_cfg["vectorstore"]["embedding_model"] == "all-MiniLM-L6-v2"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_config_merging_overrides_code_index(self, *_mocks):
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        hive = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)

        assert hive._merged_cfg["code_index"]["collection_name"] == "bms_code"
        assert hive._merged_cfg["code_index"]["sync_paths"] == ["src/BMS"]

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_config_isolation_between_hives(self, *_mocks):
        """Two hives created from the same global config must not share state."""
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        bms = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)
        common = Hive("common", cfg["hives"]["definitions"]["common"], cfg)

        # Different docs_dir
        assert bms._merged_cfg["paths"]["docs_dir"] == "docs/agentKT"
        assert common._merged_cfg["paths"]["docs_dir"] == "docs/common"

        # Different vectorstore collections
        assert bms._merged_cfg["vectorstore"]["collection_name"] == "bms_docs"
        assert common._merged_cfg["vectorstore"]["collection_name"] == "common_docs"

        # Different code index
        assert bms._merged_cfg["code_index"]["collection_name"] == "bms_code"
        assert common._merged_cfg["code_index"]["collection_name"] == "common_code"

        # Original global config is NOT mutated
        assert cfg["vectorstore"]["collection_name"] == "default_docs"
        assert cfg["paths"]["docs_dir"] == "docs/default"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_hive_sets_brain_context(self, *_mocks):
        """Hive should call set_hive_context on its BrainAgent."""
        from brain_ai.hive.hive import Hive

        cfg = _global_cfg()
        hive = Hive("bms", cfg["hives"]["definitions"]["bms"], cfg)

        assert hive.brain.hive_name == "bms"
        assert hive.brain.hive_display_name == "Backup Management Service"
        assert "common" in hive.brain._peer_hive_names
        assert "protection" in hive.brain._peer_hive_names
        assert "bms" not in hive.brain._peer_hive_names  # self excluded

    def test_topic_matching_hit(self):
        from brain_ai.hive.hive import Hive

        hive = MagicMock(spec=Hive)
        hive.topics = ["backup management", "DPP", "RSV"]
        hive.matches_topic = Hive.matches_topic.__get__(hive)

        score = hive.matches_topic("How does DPP backup management work?")
        assert score > 0

    def test_topic_matching_miss(self):
        from brain_ai.hive.hive import Hive

        hive = MagicMock(spec=Hive)
        hive.topics = ["backup management", "DPP", "RSV"]
        hive.matches_topic = Hive.matches_topic.__get__(hive)

        score = hive.matches_topic("What is the weather today?")
        assert score == 0.0

    def test_topic_matching_empty(self):
        from brain_ai.hive.hive import Hive

        hive = MagicMock(spec=Hive)
        hive.topics = []
        hive.matches_topic = Hive.matches_topic.__get__(hive)

        score = hive.matches_topic("anything")
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════
# Delegation signal parsing
# ══════════════════════════════════════════════════════════════════════


class TestDelegation:
    """Tests for the [DELEGATE:<hive>] protocol."""

    def test_extract_delegation_found(self):
        from brain_ai.hive.hive import Hive

        response = (
            "The backup failed in the snapshot phase. "
            "[DELEGATE:protection] Investigate VM snapshot failure for job abc-123."
        )
        result = Hive.extract_delegation(response)
        assert result is not None
        assert result["target_hive"] == "protection"
        assert "VM snapshot failure" in result["context"]

    def test_extract_delegation_not_found(self):
        from brain_ai.hive.hive import Hive

        response = "The backup policy uses incremental snapshots for efficiency."
        result = Hive.extract_delegation(response)
        assert result is None

    def test_extract_delegation_case_insensitive(self):
        from brain_ai.hive.hive import Hive

        response = "Need more info. [DELEGATE:Common] Check SDK retry behavior."
        result = Hive.extract_delegation(response)
        assert result is not None
        assert result["target_hive"] == "common"

    def test_extract_delegation_no_closing_bracket(self):
        from brain_ai.hive.hive import Hive

        response = "Partial signal [DELEGATE:common without bracket"
        result = Hive.extract_delegation(response)
        assert result is None

    def test_extract_delegation_empty_context(self):
        from brain_ai.hive.hive import Hive

        response = "Something [DELEGATE:protection]"
        result = Hive.extract_delegation(response)
        assert result is not None
        assert result["target_hive"] == "protection"
        assert result["context"] == ""


# ══════════════════════════════════════════════════════════════════════
# HiveRegistry tests
# ══════════════════════════════════════════════════════════════════════


class TestHiveRegistry:
    """Tests for brain_ai.hive.registry.HiveRegistry."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_creates_all_hives(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        assert len(registry) == 3
        assert "bms" in registry
        assert "common" in registry
        assert "protection" in registry

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_default_hive(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        assert registry.default_hive is not None
        assert registry.default_hive.name == "bms"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_get(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        hive = registry.get("common")
        assert hive is not None
        assert hive.name == "common"
        assert registry.get("nonexistent") is None

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_names(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        assert set(registry.names) == {"bms", "common", "protection"}

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_scope_summary(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        summary = registry.scope_summary()
        assert "bms" in summary
        assert "common" in summary
        assert "protection" in summary
        assert "Backup Management" in summary

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_iteration(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        registry = HiveRegistry(_global_cfg())
        hive_names = [h.name for h in registry]
        assert len(hive_names) == 3

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_registry_empty_definitions(self, *_mocks):
        from brain_ai.hive.registry import HiveRegistry

        cfg = _global_cfg()
        cfg["hives"]["definitions"] = {}
        registry = HiveRegistry(cfg)
        assert len(registry) == 0
        assert registry.default_hive is None

    def test_registry_handles_no_hives_key(self):
        from brain_ai.hive.registry import HiveRegistry

        cfg = {"llm": {"model": "test"}}
        registry = HiveRegistry(cfg)
        assert len(registry) == 0


# ══════════════════════════════════════════════════════════════════════
# HiveRouter tests
# ══════════════════════════════════════════════════════════════════════


class TestHiveRouter:
    """Tests for brain_ai.hive.router.HiveRouter."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_router_initialization(self, *_mocks):
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())
        assert len(router.registry) == 3
        assert router._last_hive is None

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_single_hive_skips_llm_routing(self, *_mocks):
        """With only one hive, router should skip the LLM call."""
        from brain_ai.hive.router import HiveRouter

        cfg = _bms_only_cfg()
        router = HiveRouter(cfg)

        # Mock the BMS hive's brain.chat
        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": "test answer"}
        )

        result = router.chat("How does backup policy work?")
        assert result["hive"] == "bms"
        assert result["response"] == "test answer"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_multi_hive_routes_via_llm(self, mock_updater, *_mocks):
        """With multiple hives, router uses LLM to select the hive."""
        from brain_ai.hive.router import HiveRouter

        # Ensure mocked updater instances have a real pending_count
        mock_updater.return_value.pending_count = 0
        mock_updater.return_value._pending_new_doc = None

        router = HiveRouter(_global_cfg())

        # Mock the LLM to return "HIVE:common"
        router.llm.generate = MagicMock(return_value="HIVE:common")

        # Mock the common hive's brain.chat
        router.registry.get("common").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": "SDK info here"}
        )

        result = router.chat("What SDK utilities are available?")
        assert result["hive"] == "common"
        assert result["agent"] == "knowledge"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_router_falls_back_to_default(self, *_mocks):
        """If LLM returns garbage, router falls back to default hive."""
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())
        router.llm.generate = MagicMock(return_value="I don't know")

        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": "fallback"}
        )

        result = router.chat("random question")
        assert result["hive"] == "bms"  # default

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_router_unknown_hive_falls_back(self, *_mocks):
        """If LLM selects a hive that doesn't exist, fall back to default."""
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())
        router.llm.generate = MagicMock(return_value="HIVE:nonexistent")

        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": "default answer"}
        )

        result = router.chat("test question")
        assert result["hive"] == "bms"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_conversation_history_tracking(self, *_mocks):
        from brain_ai.hive.router import HiveRouter

        cfg = _bms_only_cfg()
        router = HiveRouter(cfg)

        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "debug", "response": "debug response"}
        )

        router.chat("debug this error")
        assert router._last_hive == "bms"
        assert len(router._conversation_history) == 2

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_reset_conversation_clears_all(self, *_mocks):
        from brain_ai.hive.router import HiveRouter

        cfg = _bms_only_cfg()
        router = HiveRouter(cfg)

        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": "test"}
        )

        router.chat("hello")
        assert router._last_hive == "bms"

        router.reset_conversation()
        assert router._last_hive is None
        assert len(router._conversation_history) == 0


# ══════════════════════════════════════════════════════════════════════
# Cross-hive delegation tests
# ══════════════════════════════════════════════════════════════════════


class TestCrossHiveDelegation:
    """Tests for cross-hive delegation and synthesis."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_delegation_triggers_cross_hive_call(self, *_mocks):
        """When BMS agent responds with [DELEGATE:protection], router
        should dispatch to the protection hive and synthesize."""
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())

        # BMS returns a delegation signal
        bms_response = (
            "The backup job failed during VM snapshot phase. "
            "[DELEGATE:protection] Investigate VM snapshot failure for job abc-123."
        )
        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "debug", "response": bms_response}
        )

        # Protection provides the follow-up answer
        router.registry.get("protection").brain.chat = MagicMock(
            return_value={
                "agent": "debug",
                "response": "The VM was deallocated at snapshot time.",
            }
        )

        # Synthesis LLM call
        router.llm.generate = MagicMock(
            side_effect=lambda **kwargs: (
                "HIVE:bms"
                if "HIVE:" in (kwargs.get("system", "") + kwargs.get("message", ""))
                else "The backup failed because the VM was deallocated during snapshot."
            )
        )

        result = router.chat("Why did backup job abc-123 fail?")

        # Should have delegated
        assert result.get("delegated_to") == "protection"
        assert len(result.get("delegation_chain", [])) == 1
        assert result["delegation_chain"][0]["from"] == "bms"
        assert result["delegation_chain"][0]["to"] == "protection"

        # Protection hive should have been called
        router.registry.get("protection").brain.chat.assert_called_once()

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_delegation_to_unknown_hive_ignored(self, *_mocks):
        """Delegation to a non-existent hive should be ignored gracefully."""
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())

        bms_response = (
            "Here is some info. "
            "[DELEGATE:nonexistent] Need data from unknown domain."
        )
        router.registry.get("bms").brain.chat = MagicMock(
            return_value={"agent": "knowledge", "response": bms_response}
        )
        router.llm.generate = MagicMock(return_value="HIVE:bms")

        result = router.chat("test")

        # Should NOT have delegated
        assert result.get("delegated_to") is None
        # Delegation signal should be stripped
        assert "[DELEGATE:" not in result["response"]
        assert "Here is some info." in result["response"]

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_no_delegation_when_not_present(self, *_mocks):
        """Normal response without delegation signal should pass through."""
        from brain_ai.hive.router import HiveRouter

        cfg = _bms_only_cfg()
        router = HiveRouter(cfg)

        router.registry.get("bms").brain.chat = MagicMock(
            return_value={
                "agent": "knowledge",
                "response": "Backup policy uses incremental snapshots.",
            }
        )

        result = router.chat("How does backup policy work?")

        assert result.get("delegated_to") is None
        assert "incremental snapshots" in result["response"]

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_max_delegation_depth(self, *_mocks):
        """Delegation should stop at MAX_DELEGATION_DEPTH to prevent loops."""
        from brain_ai.hive.router import HiveRouter, MAX_DELEGATION_DEPTH
        from brain_ai.hive.hive import Hive

        router = HiveRouter(_global_cfg())

        # BMS delegates to protection, protection delegates to common,
        # common delegates back to BMS (loop!)
        router.registry.get("bms").brain.chat = MagicMock(
            return_value={
                "agent": "debug",
                "response": "BMS result [DELEGATE:protection] need prot info",
            }
        )
        router.registry.get("protection").brain.chat = MagicMock(
            return_value={
                "agent": "debug",
                "response": "Prot result [DELEGATE:common] need common info",
            }
        )
        router.registry.get("common").brain.chat = MagicMock(
            return_value={
                "agent": "coder",
                "response": "Common result [DELEGATE:bms] need bms info again",
            }
        )

        # Mock synthesis to return a simple combined text
        router.llm.generate = MagicMock(
            side_effect=lambda **kwargs: (
                "HIVE:bms"
                if "HIVE:" in kwargs.get("message", "")
                else "Synthesized answer from multiple domains."
            )
        )

        result = router.chat("complex cross-cutting question")

        # Should have stopped due to max depth (3) — not infinite loop
        chain = result.get("delegation_chain", [])
        assert len(chain) <= MAX_DELEGATION_DEPTH


# ══════════════════════════════════════════════════════════════════════
# BrainAgent hive context tests
# ══════════════════════════════════════════════════════════════════════


class TestBrainAgentHiveContext:
    """Tests for BrainAgent's hive-awareness methods."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_set_hive_context(self, *_mocks):
        from brain_ai.agents.brain_agent import BrainAgent

        cfg = _global_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        brain.set_hive_context("bms", "Backup Management Service", ["bms", "common", "protection"])

        assert brain.hive_name == "bms"
        assert brain.hive_display_name == "Backup Management Service"
        assert "common" in brain._peer_hive_names
        assert "protection" in brain._peer_hive_names
        assert "bms" not in brain._peer_hive_names  # self excluded

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_delegation_hint_with_peers(self, *_mocks):
        from brain_ai.agents.brain_agent import BrainAgent

        cfg = _global_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        brain.set_hive_context("bms", "BMS", ["bms", "common"])

        hint = brain._build_delegation_hint()
        assert "CROSS-DOMAIN DELEGATION" in hint
        assert "common" in hint
        assert "[DELEGATE:" in hint

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_delegation_hint_no_peers(self, *_mocks):
        from brain_ai.agents.brain_agent import BrainAgent

        cfg = _global_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        # No hive context set — no peers

        hint = brain._build_delegation_hint()
        assert hint == ""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_delegation_hint_single_hive(self, *_mocks):
        from brain_ai.agents.brain_agent import BrainAgent

        cfg = _global_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        brain.set_hive_context("bms", "BMS", ["bms"])  # only self

        hint = brain._build_delegation_hint()
        assert hint == ""


# ══════════════════════════════════════════════════════════════════════
# Backward compatibility
# ══════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure hives.enabled: false keeps the old single-BrainAgent behavior."""

    def test_config_template_hives_disabled_by_default(self):
        """config.yaml.template should have hives.enabled: false."""
        from brain_ai.config import load_config, reset_config

        from pathlib import Path

        reset_config()
        cfg = load_config(
            Path(__file__).resolve().parent.parent / "config.yaml.template"
        )
        assert cfg.get("hives", {}).get("enabled") is False
        reset_config()

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_brain_agent_works_without_hive_context(self, *_mocks):
        """BrainAgent should work fine without any hive context set."""
        from brain_ai.agents.brain_agent import BrainAgent

        cfg = _global_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        assert brain.hive_name is None
        assert brain._peer_hive_names == []
        assert brain._build_delegation_hint() == ""


# ══════════════════════════════════════════════════════════════════════
# Router prompt tests
# ══════════════════════════════════════════════════════════════════════


class TestHiveRouterPrompt:
    """Tests for the HiveRouter's system prompt."""

    def test_prompt_contains_hive_keyword(self):
        from brain_ai.hive.router import HIVE_ROUTER_SYSTEM_PROMPT

        assert "HIVE:" in HIVE_ROUTER_SYSTEM_PROMPT
        assert "{hive_scope_summary}" in HIVE_ROUTER_SYSTEM_PROMPT
        assert "{default_hive}" in HIVE_ROUTER_SYSTEM_PROMPT

    def test_synthesis_prompt_structure(self):
        from brain_ai.hive.router import SYNTHESIS_PROMPT

        assert "{question}" in SYNTHESIS_PROMPT
        assert "{primary_hive}" in SYNTHESIS_PROMPT
        assert "{delegated_hive}" in SYNTHESIS_PROMPT


# ══════════════════════════════════════════════════════════════════════
# End-to-end demo scenario
# ══════════════════════════════════════════════════════════════════════


class TestEndToEndScenario:
    """Simulate a realistic cross-hive debugging scenario."""

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_backup_failure_cross_hive_scenario(self, *_mocks):
        """
        Scenario: User asks about a backup job failure.
        1. Router → BMS hive (backup is BMS domain)
        2. BMS debug agent finds error is in Protection layer → delegates
        3. Protection hive investigates → finds VM was deallocated
        4. Router synthesizes: "VM was deallocated during snapshot"
        """
        from brain_ai.hive.router import HiveRouter

        router = HiveRouter(_global_cfg())

        # BMS hive: debug agent finds protection layer issue
        router.registry.get("bms").brain.chat = MagicMock(
            return_value={
                "agent": "debug",
                "response": (
                    "The backup job abc-123 failed with UserErrorVmNotFound "
                    "during the VM snapshot phase. The error originates from "
                    "the Protection layer.\n\n"
                    "[DELEGATE:protection] Investigate VM snapshot failure: "
                    "job abc-123, error UserErrorVmNotFound, occurred during "
                    "snapshot phase at 2026-05-01T10:00:00Z."
                ),
            }
        )

        # Protection hive: finds the VM was deallocated
        router.registry.get("protection").brain.chat = MagicMock(
            return_value={
                "agent": "debug",
                "response": (
                    "The VM associated with job abc-123 was in a deallocated "
                    "state at the time of the snapshot attempt. The workload "
                    "agent attempted 3 retries but the VM did not come back "
                    "online. Recommendation: ensure the VM is running before "
                    "the backup window or configure pre-backup scripts."
                ),
            }
        )

        # Mock the LLM for both routing (tiebreaker) and synthesis
        call_count = [0]

        def mock_generate(**kwargs):
            call_count[0] += 1
            msg = kwargs.get("message", "")
            sys = kwargs.get("system", "")
            # Gateway LLM tiebreaker
            if "routing classifier" in sys.lower() or "HIVE:" in sys:
                return "HIVE:bms"
            # Synthesis call
            return (
                "## Root Cause Analysis\n\n"
                "The backup job `abc-123` failed because the VM was in a "
                "**deallocated state** during the snapshot phase. The "
                "Protection agent attempted 3 retries, but the VM did not "
                "come back online.\n\n"
                "### Recommendation\n"
                "Ensure the VM is running before the backup window or "
                "configure pre-backup scripts."
            )

        router.llm.generate = MagicMock(side_effect=mock_generate)

        result = router.chat(
            "A backup job for SQL on Azure VM failed with error code "
            "UserErrorVmNotFound. The job ID is abc-123. What went wrong?"
        )

        # Verify the full flow
        assert result["hive"] == "bms"
        assert result["delegated_to"] == "protection"
        assert "deallocated" in result["response"]
        assert "abc-123" in result["response"]
        assert len(result["delegation_chain"]) == 1
        assert result["delegation_chain"][0] == {
            "from": "bms",
            "to": "protection",
            "context": result["delegation_chain"][0]["context"],  # truncated
        }

        # Both hives should have been called
        router.registry.get("bms").brain.chat.assert_called_once()
        router.registry.get("protection").brain.chat.assert_called_once()
