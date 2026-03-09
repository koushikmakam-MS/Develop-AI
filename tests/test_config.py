"""Tests for brain_ai.config — deep merge, load/cache, env overrides."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from brain_ai.config import _deep_merge, load_config, reset_config

# ── _deep_merge ──────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_flat_keys_overwrite(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_dicts_merge(self):
        base = {"llm": {"model": "gpt-4", "max_tokens": 4096}}
        override = {"llm": {"api_key": "secret"}}
        result = _deep_merge(base, override)
        assert result["llm"]["model"] == "gpt-4"          # preserved
        assert result["llm"]["max_tokens"] == 4096         # preserved
        assert result["llm"]["api_key"] == "secret"        # added

    def test_nested_override_value(self):
        base = {"llm": {"model": "gpt-4"}}
        override = {"llm": {"model": "gpt-4o"}}
        result = _deep_merge(base, override)
        assert result["llm"]["model"] == "gpt-4o"

    def test_override_adds_new_section(self):
        base = {"a": 1}
        override = {"b": {"nested": True}}
        result = _deep_merge(base, override)
        assert result["b"]["nested"] is True

    def test_override_replaces_non_dict_with_dict(self):
        """If base has a scalar and override has a dict, override wins."""
        base = {"x": "string"}
        override = {"x": {"now": "a dict"}}
        result = _deep_merge(base, override)
        assert result["x"] == {"now": "a dict"}

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deeply_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99, "e": 3}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2, "e": 3}}}


# ── load_config ──────────────────────────────────────────────────────────

class TestLoadConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_loads_template_successfully(self):
        template = Path(__file__).resolve().parent.parent / "config.yaml.template"
        cfg = load_config(template)
        assert "llm" in cfg
        assert "azure_devops" in cfg
        assert "kusto" in cfg

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")

    def test_caching_returns_same_object(self):
        template = Path(__file__).resolve().parent.parent / "config.yaml.template"
        cfg1 = load_config(template)
        cfg2 = load_config(template)
        assert cfg1 is cfg2  # same cached reference

    def test_reset_clears_cache(self):
        template = Path(__file__).resolve().parent.parent / "config.yaml.template"
        cfg1 = load_config(template)
        reset_config()
        cfg2 = load_config(template)
        assert cfg1 is not cfg2  # new dict after reset

    def test_env_var_overrides_pat(self, tmp_path):
        """Environment variable should override the config file value."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "azure_devops": {
                "pat": "from-file",
                "repo_url": "https://x.visualstudio.com/P/_git/R",
                "branch": "main",
            },
            "llm": {"model": "m", "endpoint": "e", "api_key": "k"},
            "kusto": {"cluster_url": "c", "database": "d"},
            "vectorstore": {"persist_dir": ".chromadb"},
            "paths": {"docs_dir": "docs", "repo_clone_dir": ".repo_cache"},
            "agents": {"enabled": ["knowledge"]},
        }))
        reset_config()
        with patch.dict(os.environ, {"BCDR_DEVAI_AZURE_DEVOPS_PAT": "from-env"}):
            cfg = load_config(config_file)
            assert cfg["azure_devops"]["pat"] == "from-env"

    def test_local_yaml_merges(self, tmp_path):
        """config.local.yaml should merge on top of config.yaml."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "llm": {"model": "gpt-4", "endpoint": "e", "api_key": "base-key"},
            "azure_devops": {"pat": "p", "repo_url": "u", "branch": "b"},
            "kusto": {"cluster_url": "c", "database": "d"},
            "vectorstore": {"persist_dir": ".chromadb"},
            "paths": {"docs_dir": "docs", "repo_clone_dir": ".repo_cache"},
            "agents": {"enabled": ["knowledge"]},
        }))
        local_file = tmp_path / "config.local.yaml"
        local_file.write_text(yaml.dump({
            "llm": {"api_key": "local-secret"},
        }))
        reset_config()
        cfg = load_config(config_file)
        assert cfg["llm"]["api_key"] == "local-secret"
        assert cfg["llm"]["model"] == "gpt-4"  # preserved from base
