#!/usr/bin/env python3
"""Quick quality test — send diverse queries through the HiveRouter and print results."""

import sys, time

from brain_ai.config import get_config
from brain_ai.hive.router import HiveRouter

cfg = get_config()
router = HiveRouter(cfg)

QUERIES = [
    # 1. BMS-specific
    "How does the backup policy scheduling work? Show me the key classes involved.",
    # 2. Dataplane-specific
    "How does the data plane handle restore point expiry and garbage collection?",
    # 3. Common library
    "What serialization utilities does the Common library provide?",
    # 4. Cross-hive (should trigger proactive discovery)
    "Walk me through the end-to-end flow when a VM backup job is triggered — from BMS to data plane.",
    # 5. Protection / workload coordination
    "How does the workload coordination service handle SQL quiesce before snapshot?",
]

for i, q in enumerate(QUERIES, 1):
    print(f"\n{'='*80}")
    print(f"  Q{i}: {q}")
    print(f"{'='*80}")
    t0 = time.time()
    result = router.chat(q, deep=True)
    elapsed = time.time() - t0

    hive   = result.get("routed_to", "?")
    answer = result.get("response", "")
    xhive  = result.get("cross_hive_sources", [])

    # Quality metrics
    length = len(answer)
    has_code = "```" in answer or "`" in answer
    has_headers = "##" in answer or "**" in answer

    print(f"  Routed to : {hive}")
    if xhive:
        print(f"  Cross-hive: {xhive}")
    print(f"  Length    : {length:,} chars  |  Code refs: {has_code}  |  Headers: {has_headers}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"\n{answer[:1500]}")
    if length > 1500:
        print(f"\n  ... ({length - 1500:,} more chars)")
    print()
