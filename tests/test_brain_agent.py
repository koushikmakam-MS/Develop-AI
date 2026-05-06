"""Tests for brain_ai.agents.brain_agent — routing, thresholds, fallback."""

import json
import os
from unittest.mock import MagicMock, patch

from brain_ai.agents.brain_agent import ROUTER_SYSTEM_PROMPT, BrainAgent


def _make_cfg():
    """Minimal config with all agents disabled (we mock them individually)."""
    return {
        "llm": {
            "model": "test",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "key",
        },
        "vectorstore": {
            "persist_dir": ".chromadb",
            "collection": "test",
            "code_collection": "test_code",
        },
        "paths": {
            "docs_dir": "docs/agentKT",
            "repo_clone_dir": ".repo_cache",
        },
        "kusto": {
            "cluster_url": "https://c",
            "database": "db",
            "mcp_url": "http://127.0.0.1:8701",
        },
        "azure_devops": {
            "pat": "p",
            "repo_url": "https://x.visualstudio.com/P/_git/R",
            "branch": "main",
        },
        "agents": {
            "enabled": [],  # we'll register mocks manually
            "knowledge_confidence_threshold": 0.45,
            "doc_gap_threshold": 0.50,
        },
    }


# ── Router system prompt ─────────────────────────────────────────────────

class TestRouterPrompt:
    def test_all_agents_mentioned(self):
        for agent in ["knowledge", "debug", "coder", "knowledge_updater"]:
            assert agent in ROUTER_SYSTEM_PROMPT

    def test_route_format_mentioned(self):
        assert "ROUTE:" in ROUTER_SYSTEM_PROMPT


# ── Threshold properties ─────────────────────────────────────────────────

class TestThresholds:
    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_threshold_from_config(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        assert brain._knowledge_confidence_threshold == 0.45
        assert brain._doc_gap_threshold == 0.50

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_threshold_defaults(self, *_mocks):
        cfg = _make_cfg()
        del cfg["agents"]["knowledge_confidence_threshold"]
        del cfg["agents"]["doc_gap_threshold"]
        brain = BrainAgent(cfg)
        assert brain._knowledge_confidence_threshold == 0.35
        assert brain._doc_gap_threshold == 0.50


# ── Routing ──────────────────────────────────────────────────────────────

class TestRouting:
    def _make_brain(self):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        return brain

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_parses_knowledge(self, *_mocks):
        brain = self._make_brain()
        brain._agents["knowledge"] = MagicMock()
        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge")
        assert brain._route("How does backup work?") == "knowledge"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_parses_debug(self, *_mocks):
        brain = self._make_brain()
        brain._agents["debug"] = MagicMock()
        brain.llm.generate = MagicMock(return_value="ROUTE:debug")
        assert brain._route("Debug this TaskId") == "debug"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_parses_coder(self, *_mocks):
        brain = self._make_brain()
        brain._agents["coder"] = MagicMock()
        brain.llm.generate = MagicMock(return_value="ROUTE:coder")
        assert brain._route("trace the code") == "coder"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_defaults_on_garbage(self, *_mocks):
        brain = self._make_brain()
        brain.llm.generate = MagicMock(return_value="I don't know what to do")
        assert brain._route("hello") == "knowledge"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_defaults_on_exception(self, *_mocks):
        brain = self._make_brain()
        brain.llm.generate = MagicMock(side_effect=RuntimeError("boom"))
        assert brain._route("hello") == "knowledge"

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_injects_pending_corrections_state(self, *_mocks):
        """When there are pending corrections, the router system prompt includes state info."""
        brain = self._make_brain()
        mock_updater = MagicMock()
        mock_updater.pending_count = 2
        mock_updater._pending_new_doc = None
        brain._agents["knowledge_updater"] = mock_updater
        brain._agents["knowledge"] = MagicMock()
        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge_updater")

        brain._route("submit")

        # Inspect the system prompt passed to the LLM
        call_kwargs = brain.llm.generate.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system", "")
        assert "2 pending correction(s)" in system_prompt

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_route_injects_pending_new_doc_state(self, *_mocks):
        """When there's a pending new-doc offer, the router sees that state."""
        brain = self._make_brain()
        mock_updater = MagicMock()
        mock_updater.pending_count = 0
        mock_updater._pending_new_doc = {"topic": "test", "corrections": ["c"], "search_query": "s"}
        brain._agents["knowledge_updater"] = mock_updater
        brain._agents["knowledge"] = MagicMock()
        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge_updater")

        brain._route("create doc")

        call_kwargs = brain.llm.generate.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system", "")
        assert "offered to create a new document" in system_prompt


# ── Knowledge → Coder fallback ───────────────────────────────────────────

class TestKnowledgeCoderFallback:
    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_low_confidence_falls_back_to_coder(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        # Mock knowledge agent with low confidence
        mock_knowledge = MagicMock()
        mock_knowledge.answer_with_confidence.return_value = ("weak doc answer", 0.20)
        brain._agents["knowledge"] = mock_knowledge

        # Mock coder agent with a good answer
        mock_coder = MagicMock()
        mock_coder.analyze_with_boundaries.return_value = ("Here's the code path...", [])
        brain._agents["coder"] = mock_coder

        # Mock LLM for synthesis
        brain.llm.generate = MagicMock(side_effect=[
            "ROUTE:knowledge",              # routing call
            "Synthesized doc+code answer",   # synthesis call
        ])

        result = brain.chat("some question")
        assert result["agent"] == "knowledge+coder"
        mock_coder.analyze_with_boundaries.assert_called_once()

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_high_confidence_stays_with_knowledge(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        mock_knowledge = MagicMock()
        mock_knowledge.answer_with_confidence.return_value = ("great doc answer", 0.85)
        brain._agents["knowledge"] = mock_knowledge

        # No coder agent available → stays with knowledge only
        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge")

        result = brain.chat("well-documented question")
        assert result["agent"] == "knowledge"
        assert "great doc answer" in result["response"]

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_coder_fallback_with_no_code_returns_helpful_message(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        mock_knowledge = MagicMock()
        mock_knowledge.answer_with_confidence.return_value = ("weak", 0.10)
        brain._agents["knowledge"] = mock_knowledge

        mock_coder = MagicMock()
        mock_coder.analyze_with_boundaries.return_value = ("I don't have any indexed source code yet.", [])
        brain._agents["coder"] = mock_coder

        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge")

        result = brain.chat("obscure topic")
        assert "couldn't find strong matches" in result["response"]


# ── Doc gap logging ──────────────────────────────────────────────────────

class TestDocGapLogging:
    def test_doc_gap_logged_below_threshold(self, tmp_path):
        with patch("brain_ai.agents.brain_agent.KnowledgeAgent"), \
             patch("brain_ai.agents.brain_agent.DebugAgent"), \
             patch("brain_ai.agents.brain_agent.CoderAgent"), \
             patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent"):
            cfg = _make_cfg()
            cfg["agents"]["enabled"] = []
            cfg["paths"]["docs_dir"] = str(tmp_path / "docs")
            os.makedirs(cfg["paths"]["docs_dir"], exist_ok=True)
            brain = BrainAgent(cfg)

            brain._log_doc_gap("some undocumented question", 0.25)

            gap_file = tmp_path / "doc_gaps.json"
            assert gap_file.exists()
            gaps = json.loads(gap_file.read_text())
            assert len(gaps) == 1
            assert gaps[0]["question"] == "some undocumented question"
            assert gaps[0]["confidence"] == 0.25

    def test_doc_gap_appends_to_existing(self, tmp_path):
        with patch("brain_ai.agents.brain_agent.KnowledgeAgent"), \
             patch("brain_ai.agents.brain_agent.DebugAgent"), \
             patch("brain_ai.agents.brain_agent.CoderAgent"), \
             patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent"):
            cfg = _make_cfg()
            cfg["agents"]["enabled"] = []
            cfg["paths"]["docs_dir"] = str(tmp_path / "docs")
            os.makedirs(cfg["paths"]["docs_dir"], exist_ok=True)

            gap_file = tmp_path / "doc_gaps.json"
            gap_file.write_text(json.dumps([{"question": "old", "confidence": 0.1}]))

            brain = BrainAgent(cfg)
            brain._log_doc_gap("new question", 0.30)

            gaps = json.loads(gap_file.read_text())
            assert len(gaps) == 2


# ── Conversation history ─────────────────────────────────────────────────

class TestConversationHistory:
    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_history_capped_at_20(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)

        mock_agent = MagicMock()
        mock_agent.answer_with_confidence.return_value = ("ok", 0.90)
        brain._agents["knowledge"] = mock_agent
        brain.llm.generate = MagicMock(return_value="ROUTE:knowledge")

        for i in range(15):
            brain.chat(f"message {i}")

        assert len(brain._conversation_history) <= 20

    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    def test_reset_clears_history(self, *_mocks):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = []
        brain = BrainAgent(cfg)
        brain._conversation_history = [{"role": "user", "content": "hi"}]
        brain._last_agent = "knowledge"
        brain.reset_conversation()
        assert brain._conversation_history == []
        assert brain._last_agent is None


# ── Debug Agent Code Enrichment ─────────────────────────────────────────

class TestDebugCodeEnrichment:
    """Tests for debug agent code reference extraction and coder wiring."""

    def test_extract_code_refs_from_file_line(self):
        from brain_ai.agents.debug_agent import DebugAgent
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = ["debug"]
        agent = DebugAgent.__new__(DebugAgent)
        # Test file:line extraction
        text = "Error in BMSHandler.cs:245 and DataProtectionResource.cs(312)"
        refs = agent._extract_code_refs(text)
        assert "BMSHandler" in refs
        assert "DataProtectionResource" in refs

    def test_extract_code_refs_from_class_method(self):
        from brain_ai.agents.debug_agent import DebugAgent
        agent = DebugAgent.__new__(DebugAgent)
        text = "Failed at BackupController.ValidateRequest during processing"
        refs = agent._extract_code_refs(text)
        assert "BackupController.ValidateRequest" in refs

    def test_extract_code_refs_skips_system_namespaces(self):
        from brain_ai.agents.debug_agent import DebugAgent
        agent = DebugAgent.__new__(DebugAgent)
        text = "System.Net.Http.HttpClient threw Microsoft.Azure.Storage.Exception"
        refs = agent._extract_code_refs(text)
        # Should skip System.* and Microsoft.Azure.* prefixed
        for r in refs:
            assert not r.startswith("System.")
            assert not r.startswith("Microsoft.Azure.")

    def test_extract_code_refs_roles(self):
        from brain_ai.agents.debug_agent import DebugAgent
        agent = DebugAgent.__new__(DebugAgent)
        text = "Role: BMSWebRole logged error, BMSDppTeeWorkerRole also affected"
        refs = agent._extract_code_refs(text)
        assert "BMSWebRole" in refs
        assert "BMSDppTeeWorkerRole" in refs

    def test_set_coder_wires_agent(self):
        from brain_ai.agents.debug_agent import DebugAgent
        agent = DebugAgent.__new__(DebugAgent)
        agent._coder = None
        mock_coder = MagicMock()
        agent.set_coder(mock_coder)
        assert agent._coder is mock_coder

    @patch("brain_ai.agents.brain_agent.KnowledgeUpdaterAgent")
    @patch("brain_ai.agents.brain_agent.CoderAgent")
    @patch("brain_ai.agents.brain_agent.DebugAgent")
    @patch("brain_ai.agents.brain_agent.KnowledgeAgent")
    def test_brain_wires_coder_into_debug(self, MockKnowledge, MockDebug, MockCoder, MockUpdater):
        cfg = _make_cfg()
        cfg["agents"]["enabled"] = ["debug", "coder"]
        brain = BrainAgent(cfg)
        # Verify set_coder was called on the debug agent with the coder agent
        MockDebug.return_value.set_coder.assert_called_once_with(MockCoder.return_value)

    def test_get_code_context_returns_none_without_coder(self):
        from brain_ai.agents.debug_agent import DebugAgent
        agent = DebugAgent.__new__(DebugAgent)
        agent._coder = None
        result = agent._get_code_context("some summary", "some issue")
        assert result is None
