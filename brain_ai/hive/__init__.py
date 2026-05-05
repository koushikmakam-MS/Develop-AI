"""
🐝 Agent Hive Architecture — domain-specialized agent clusters.

Each Hive is a self-contained domain with its own BrainAgent, docs,
code index, and vector collections.  The HiveRouter sits above all
hives and dispatches user queries to the appropriate domain.
"""

from brain_ai.hive.hive import Hive
from brain_ai.hive.registry import HiveRegistry
from brain_ai.hive.router import HiveRouter

__all__ = ["Hive", "HiveRegistry", "HiveRouter"]
