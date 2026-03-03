"""Basic smoke tests for BCDR DeveloperAI modules."""

import pytest
from pathlib import Path


def test_config_loads_template():
    """Verify config loader reads the template without crashing."""
    from brain_ai.config import load_config, reset_config

    reset_config()
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml.template")
    assert "azure_devops" in cfg
    assert "llm" in cfg
    assert "kusto" in cfg
    assert "vectorstore" in cfg
    assert "paths" in cfg
    assert "agents" in cfg
    reset_config()


def test_chunk_text():
    """Verify the markdown chunker splits correctly."""
    from brain_ai.vectorstore.indexer import _chunk_text

    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = _chunk_text(text, chunk_size=50, overlap=0)
    assert len(chunks) >= 1
    assert "Paragraph one" in chunks[0]


def test_chunk_text_overlap():
    """Chunks with overlap should carry trailing context."""
    from brain_ai.vectorstore.indexer import _chunk_text

    paragraphs = [f"Paragraph {i} with some content." for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = _chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 1


def test_agent_registry_keys():
    """Verify brain agent recognises the expected agent names."""
    from brain_ai.agents.brain_agent import ROUTER_SYSTEM_PROMPT

    assert "knowledge" in ROUTER_SYSTEM_PROMPT
    assert "debug" in ROUTER_SYSTEM_PROMPT


def test_llm_client_import():
    """Verify LLMClient can be imported."""
    from brain_ai.llm_client import LLMClient

    assert LLMClient is not None
