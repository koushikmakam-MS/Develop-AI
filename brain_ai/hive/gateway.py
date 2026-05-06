"""
Gateway Agent — two-stage hive routing with confidence scoring.

Stage 1 (Fast):  Topic-based keyword matching against all hives.
                 If one hive wins by a clear margin → route directly.

Stage 2 (Smart): LLM tiebreaker with ONLY the top candidates,
                 producing a focused, accurate classification.

This replaces the single-shot LLM routing which struggles with
closely related hives (e.g. DPP vs RSV, Regional vs PIT Catalog).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from brain_ai.hive.hive import Hive
from brain_ai.hive.registry import HiveRegistry
from brain_ai.llm_client import LLMClient

log = logging.getLogger(__name__)

# If top scorer leads second by this ratio, skip LLM
CONFIDENCE_THRESHOLD = 2.0

# Max candidates to send to LLM tiebreaker
MAX_LLM_CANDIDATES = 4

TIEBREAKER_PROMPT = """You are a routing classifier for a multi-domain BCDR (Backup & Disaster Recovery) system.

The user asked:
\"{question}\"

Candidate domains (ranked by keyword overlap):
{candidates}

Pick the ONE domain that should handle this question.
Consider:
- Which domain OWNS the primary concept?
- DPP = Backup Vaults, DPP-specific backup instances/policies, cross-region restore, resource guards, Azure Arc, immutability
- RSV = Recovery Services Vault, protection containers, protected items, backup engines/fabrics, vault config, backup providers
- If the user mentions "vault" alone, it could be DPP (Backup Vault) or RSV (Recovery Services Vault) — look for other clues
- Follow-up context: {follow_up_context}

Respond with EXACTLY:
HIVE:<domain_name>
"""


class Gateway:
    """Two-stage hive routing: fast topic match → LLM tiebreaker.

    Entry points are restricted to primary hives (e.g. DPP, RSV).
    Other hives are reached via cross-hive delegation or proactive discovery.
    """

    def __init__(
        self,
        registry: HiveRegistry,
        llm: LLMClient,
        default_hive: str,
        primary_hives: Optional[List[str]] = None,
    ):
        self.registry = registry
        self.llm = llm
        self.default_hive = default_hive
        self.primary_hives: List[str] = primary_hives or [default_hive]

        # Pre-build keyword index: word → [(hive_name, topic)]
        self._keyword_index: Dict[str, List[Tuple[str, str]]] = {}
        for hive in registry:
            for topic in hive.topics:
                for word in topic.lower().split():
                    word = word.strip(".,;:()")
                    if len(word) >= 3:  # skip tiny words
                        self._keyword_index.setdefault(word, []).append(
                            (hive.name, topic)
                        )

        log.info(
            "Gateway initialized — %d hives, %d indexed keywords",
            len(registry),
            len(self._keyword_index),
        )

    def route(
        self,
        message: str,
        last_hive: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Route a message to the best hive.

        Returns dict with:
            hive: str           — selected hive name
            confidence: float   — 0.0–1.0 confidence score
            method: str         — "topic_match" or "llm_tiebreaker"
            scores: dict        — per-hive scores
            matched_topics: dict — per-hive matched topic keywords
        """
        # Stage 1: Topic scoring
        scores, matched = self._score_all(message)

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        log.info(
            "Gateway scores: %s",
            [(name, f"{score:.3f}") for name, score in ranked[:5]],
        )

        # If all scores are 0, use suffix detection / primary fallback
        if not ranked or ranked[0][1] == 0:
            choice, method = self._zero_score_fallback(message, last_hive)
            log.info("Gateway: no topic matches → %s (%s)", choice, method)
            return {
                "hive": choice,
                "confidence": 0.0,
                "method": method,
                "scores": scores,
                "matched_topics": matched,
            }

        top_name, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        # Clear winner? Route directly (but enforce primary-only).
        if second_score == 0 or (top_score / max(second_score, 0.001)) >= CONFIDENCE_THRESHOLD:
            routed = self._enforce_primary(top_name, ranked, scores, matched)
            log.info(
                "Gateway: clear winner → %s (score=%.3f, 2nd=%.3f, routed=%s)",
                top_name, top_score, second_score, routed["hive"],
            )
            return routed

        # Stage 2: LLM tiebreaker with top candidates only
        candidates = ranked[:MAX_LLM_CANDIDATES]
        return self._llm_tiebreaker(
            message, candidates, matched, scores, last_hive
        )

    def _zero_score_fallback(
        self, message: str, last_hive: Optional[str]
    ) -> Tuple[str, str]:
        """Determine the best primary hive when scoring produced no matches.

        Priority:
        1. Follow-up to a primary hive → stay there.
        2. Default to first primary hive (DPP).
        """
        if last_hive and last_hive in self.primary_hives:
            return last_hive, "primary_followup"

        return self.primary_hives[0], "primary_default"

    def _best_primary(
        self, ranked: List[Tuple[str, float]]
    ) -> Optional[Tuple[str, float]]:
        """Find the highest-scoring primary hive from the ranked list."""
        for name, score in ranked:
            if name in self.primary_hives and score > 0:
                return name, score
        return None

    def _enforce_primary(
        self,
        top_name: str,
        ranked: List[Tuple[str, float]],
        scores: Dict[str, float],
        matched: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Ensure the routed hive is a primary hive.

        If the top scorer is not primary, pick the best primary hive instead.
        If no primary hive scored, default to the first primary.
        """
        if top_name in self.primary_hives:
            top_score = scores[top_name]
            second_score = max(
                (s for n, s in ranked if n != top_name), default=0
            )
            return {
                "hive": top_name,
                "confidence": min(top_score / max(top_score + second_score, 1), 1.0),
                "method": "topic_match",
                "scores": scores,
                "matched_topics": matched,
            }

        # Top scorer is non-primary → find best primary instead
        best = self._best_primary(ranked)
        if best:
            name, score = best
            log.info(
                "Gateway: top scorer '%s' is non-primary, using primary '%s' (score=%.3f)",
                top_name, name, score,
            )
            return {
                "hive": name,
                "confidence": min(score / max(score + scores.get(top_name, 0), 1), 1.0),
                "method": "primary_override",
                "scores": scores,
                "matched_topics": matched,
            }

        # No primary hive scored at all → default
        log.info(
            "Gateway: no primary hive scored, defaulting to '%s'",
            self.primary_hives[0],
        )
        return {
            "hive": self.primary_hives[0],
            "confidence": 0.0,
            "method": "primary_default",
            "scores": scores,
            "matched_topics": matched,
        }

    def _score_all(
        self, message: str
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Score all hives against the message using topic keywords.

        Returns (scores_dict, matched_topics_dict).
        """
        msg_lower = message.lower()
        msg_words = set(re.findall(r'\b\w{3,}\b', msg_lower))

        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}

        for hive in self.registry:
            hive_score = 0.0
            hive_matched: List[str] = []

            # Anchor terms — exclusive high-confidence identifiers (weight 5.0)
            for anchor in hive.anchor_terms:
                if anchor in msg_lower:
                    hive_score += 5.0
                    hive_matched.append(f"[anchor] {anchor}")

            for topic in hive.topics:
                topic_lower = topic.lower()

                # Exact phrase match (highest value)
                if topic_lower in msg_lower:
                    hive_score += 3.0
                    hive_matched.append(f"[exact] {topic}")
                    continue

                # Word overlap match
                topic_words = set(topic_lower.split())
                overlap = topic_words & msg_words
                if overlap:
                    # Partial match — score by fraction of topic words matched
                    frac = len(overlap) / len(topic_words)
                    hive_score += frac * 1.5
                    if frac >= 0.5:
                        hive_matched.append(f"[partial] {topic}")

            scores[hive.name] = hive_score
            matched[hive.name] = hive_matched

        return scores, matched

    def _llm_tiebreaker(
        self,
        message: str,
        candidates: List[Tuple[str, float]],
        matched: Dict[str, List[str]],
        all_scores: Dict[str, float],
        last_hive: Optional[str],
    ) -> Dict[str, Any]:
        """Use LLM to break a tie between top-scoring hives."""
        candidate_lines = []
        for name, score in candidates:
            hive = self.registry.get(name)
            if hive is None:
                continue
            topics_str = ", ".join(hive.topics[:8])
            matches_str = ", ".join(matched.get(name, [])[:5])
            candidate_lines.append(
                f"**{name}** ({hive.display_name}) — score={score:.1f}\n"
                f"  Description: {hive.description}\n"
                f"  Topics: {topics_str}\n"
                f"  Matched: {matches_str}"
            )

        follow_up = (
            f"User was previously talking to '{last_hive}' hive."
            if last_hive
            else "No prior conversation context."
        )

        prompt = TIEBREAKER_PROMPT.format(
            question=message,
            candidates="\n\n".join(candidate_lines),
            follow_up_context=follow_up,
        )

        try:
            text = self.llm.generate(
                message=prompt,
                system="You are a precise routing classifier. Respond with HIVE:<name> only.",
                history=[],
            ).strip()

            m = re.search(r'\bHIVE\s*:\s*(\w+)', text, re.IGNORECASE)
            if m:
                hive_name = m.group(1).strip().lower()
                # Enforce primary: if LLM picks non-primary, override
                if hive_name in self.registry:
                    if hive_name not in self.primary_hives:
                        best = self._best_primary(
                            sorted(all_scores.items(), key=lambda x: -x[1])
                        )
                        override = best[0] if best else self.primary_hives[0]
                        log.info(
                            "Gateway LLM picked non-primary '%s', overriding to '%s'",
                            hive_name, override,
                        )
                        hive_name = override
                    log.info("Gateway LLM tiebreaker → %s", hive_name)
                    return {
                        "hive": hive_name,
                        "confidence": 0.7,
                        "method": "llm_tiebreaker",
                        "scores": all_scores,
                        "matched_topics": matched,
                    }
                log.warning("LLM tiebreaker picked unknown hive '%s'", hive_name)

            # Fallback: use top scorer
            log.warning("LLM tiebreaker format unexpected: '%s'", text[:100])
        except Exception as e:
            log.error("LLM tiebreaker failed: %s", e)

        # Fallback to best primary from stage 1
        best = self._best_primary(
            sorted(all_scores.items(), key=lambda x: -x[1])
        )
        top = best[0] if best else self.primary_hives[0]
        return {
            "hive": top,
            "confidence": 0.5,
            "method": "primary_fallback",
            "scores": all_scores,
            "matched_topics": matched,
        }
