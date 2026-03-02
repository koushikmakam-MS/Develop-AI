"""Quick test: verify doc_gap_threshold works end-to-end."""
import logging, json, os

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from brain_ai.config import get_config
from brain_ai.agents.knowledge_agent import KnowledgeAgent
from brain_ai.agents.brain_agent import BrainAgent

cfg = get_config()

threshold = float(cfg.get("agents", {}).get("knowledge_confidence_threshold", 0.35))
gap_threshold = float(cfg.get("agents", {}).get("doc_gap_threshold", 0.45))
print(f"[CONFIG] knowledge_confidence_threshold = {threshold}")
print(f"[CONFIG] doc_gap_threshold              = {gap_threshold}")

ka = KnowledgeAgent(cfg)

# Good match — should NOT trigger doc gap
hits_good = ka.indexer.search("How does backup policy work in DPP?", top_k=3)
best_good = max(h["score"] for h in hits_good) if hits_good else 0
gap_good = best_good < gap_threshold
coder_good = best_good < threshold
print(f"\n[GOOD] 'How does backup policy work in DPP?'")
print(f"  Best score: {best_good:.3f}")
print(f"  Doc gap triggered? {gap_good}  (want: False)")
print(f"  Coder fallback?    {coder_good}  (want: False)")
assert not gap_good, f"FAIL: good match {best_good:.3f} should NOT trigger doc gap at {gap_threshold}"
assert not coder_good, f"FAIL: good match {best_good:.3f} should NOT trigger coder at {threshold}"
print("  ✅ PASS")

# Mediocre match — should trigger doc gap but NOT coder fallback
hits_med = ka.indexer.search("What is the retry logic for backup operations?", top_k=3)
best_med = max(h["score"] for h in hits_med) if hits_med else 0
gap_med = best_med < gap_threshold
coder_med = best_med < threshold
print(f"\n[MEDIOCRE] 'What is the retry logic for backup operations?'")
print(f"  Best score: {best_med:.3f}")
print(f"  Doc gap triggered? {gap_med}")
print(f"  Coder fallback?    {coder_med}")

# Poor match — should trigger BOTH doc gap and coder fallback
hits_bad = ka.indexer.search("How does Kubernetes pod scheduling work?", top_k=3)
best_bad = max(h["score"] for h in hits_bad) if hits_bad else 0
gap_bad = best_bad < gap_threshold
coder_bad = best_bad < threshold
print(f"\n[POOR] 'How does Kubernetes pod scheduling work?'")
print(f"  Best score: {best_bad:.3f}")
print(f"  Doc gap triggered? {gap_bad}  (want: True)")
print(f"  Coder fallback?    {coder_bad}  (want: True)")
assert gap_bad, f"FAIL: bad match {best_bad:.3f} SHOULD trigger doc gap at {gap_threshold}"
assert coder_bad, f"FAIL: bad match {best_bad:.3f} SHOULD trigger coder at {threshold}"
print("  ✅ PASS")

# Verify _log_doc_gap writes
ba = BrainAgent(cfg)
gap_path = os.path.normpath(
    os.path.join(cfg.get("paths", {}).get("docs_dir", "docs/agentKT"), "..", "doc_gaps.json")
)
if os.path.exists(gap_path):
    os.remove(gap_path)

ba._log_doc_gap("Test: Kubernetes pod scheduling", 0.33)
assert os.path.exists(gap_path), "FAIL: doc_gaps.json not created"
with open(gap_path, "r") as f:
    gaps = json.load(f)
assert len(gaps) == 1
assert gaps[0]["confidence"] == 0.33
print(f"\n[LOG] doc_gaps.json written correctly: {gaps[0]}")
os.remove(gap_path)
print("  ✅ PASS")

print("\n" + "=" * 50)
print("✅ All doc_gap_threshold tests passed!")
print(f"   Thresholds: coder={threshold}, doc_gap={gap_threshold}")
print(f"   Good match ({best_good:.3f}) → no gap, no coder  ✅")
print(f"   Mediocre   ({best_med:.3f}) → gap={'yes' if gap_med else 'no'}, coder={'yes' if coder_med else 'no'}")
print(f"   Poor match ({best_bad:.3f}) → gap=yes, coder=yes  ✅")
