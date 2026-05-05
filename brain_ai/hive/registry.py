"""
HiveRegistry — holds all Hive instances, provides lookup by name.

Created once at startup from the ``hives.definitions`` section of config.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from brain_ai.hive.hive import Hive

log = logging.getLogger(__name__)


class HiveRegistry:
    """Registry of all active hives."""

    def __init__(self, global_cfg: Dict[str, Any]):
        self._hives: Dict[str, Hive] = {}
        self._global_cfg = global_cfg

        hives_cfg = global_cfg.get("hives", {})
        definitions = hives_cfg.get("definitions", {})
        self.default_hive_name: str = hives_cfg.get("default_hive", "bms")

        for name, hive_cfg in definitions.items():
            try:
                hive = Hive(name, hive_cfg, global_cfg)
                self._hives[name] = hive
                log.info("Registered hive: %s", name)
            except Exception as e:
                log.warning("Failed to initialize hive '%s': %s", name, e)

        if not self._hives:
            log.warning("No hives registered — the system may not work correctly.")

        log.info(
            "HiveRegistry ready — %d hive(s): %s (default: %s)",
            len(self._hives),
            list(self._hives.keys()),
            self.default_hive_name,
        )

    # ── Lookup ──────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Hive]:
        """Get a hive by name, or None."""
        return self._hives.get(name)

    @property
    def default_hive(self) -> Optional[Hive]:
        """The default hive (fallback when router can't decide)."""
        return self._hives.get(self.default_hive_name)

    @property
    def names(self) -> List[str]:
        """All registered hive names."""
        return list(self._hives.keys())

    @property
    def hives(self) -> List[Hive]:
        """All registered hive instances."""
        return list(self._hives.values())

    def __len__(self) -> int:
        return len(self._hives)

    def __iter__(self) -> Iterator[Hive]:
        return iter(self._hives.values())

    def __contains__(self, name: str) -> bool:
        return name in self._hives

    # ── Scope summary (for router prompt) ───────────────────────────

    def scope_summary(self) -> str:
        """Build a textual scope summary for the router LLM prompt.

        Returns a formatted string describing each hive's name,
        description, and topic keywords.
        """
        lines = []
        for hive in self._hives.values():
            topics_str = ", ".join(hive.topics[:10])
            if len(hive.topics) > 10:
                topics_str += f", ... (+{len(hive.topics) - 10} more)"
            lines.append(
                f'- "{hive.name}" ({hive.display_name}): '
                f"{hive.description.strip()} "
                f"Topics: [{topics_str}]"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"HiveRegistry(hives={self.names})"
