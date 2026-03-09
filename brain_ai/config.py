"""
Configuration loader for BCDR DeveloperAI.

Reads config.yaml (generic, committed) and merges config.local.yaml
(secrets/overrides, gitignored) on top.  Environment variables have
the highest priority.

Load order (last wins):
  1. config.yaml           — structural settings (committed)
  2. config.local.yaml     — secrets & overrides (gitignored)
  3. Environment variables — BCDR_DEVAI_AZURE_DEVOPS_PAT, etc.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_CACHE: Dict[str, Any] | None = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.local.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load and return the configuration dictionary.

    1. Read *config_path* (defaults to ``config.yaml``).
    2. If ``config.local.yaml`` exists next to it, deep-merge on top.
    3. Apply environment-variable overrides.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Copy config.yaml.template to config.yaml and fill in your values."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Merge config.local.yaml (secrets & overrides) if it exists
    local_path = path.parent / "config.local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, local_cfg)

    # Environment-variable overrides for secrets (highest priority)
    env_overrides = {
        ("azure_devops", "pat"): "BCDR_DEVAI_AZURE_DEVOPS_PAT",
        ("llm", "api_key"): "BCDR_DEVAI_LLM_API_KEY",
        ("kusto", "cluster_url"): "BCDR_DEVAI_KUSTO_CLUSTER_URL",
        ("kusto", "database"): "BCDR_DEVAI_KUSTO_DATABASE",
    }
    for (section, key), env_var in env_overrides.items():
        val = os.environ.get(env_var)
        if val:
            cfg.setdefault(section, {})[key] = val

    _CONFIG_CACHE = cfg
    return cfg


def get_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Convenience alias for load_config."""
    return load_config(config_path)


def reset_config():
    """Clear cached config (useful for tests)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
