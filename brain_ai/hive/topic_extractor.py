"""
Topic Extractor — auto-generates routing topics from indexed docs and code.

After a hive is indexed, this module:
  1. Samples doc titles, headers, and key content from ChromaDB
  2. Sends them to the LLM to extract domain-specific topics
  3. Stores the result in the DiscoveryStore

This keeps the router's topic-based routing always fresh
without manual config.yaml maintenance.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from brain_ai.hive.discovery_store import DiscoveryStore
from brain_ai.llm_client import LLMClient

log = logging.getLogger(__name__)

TOPIC_EXTRACTION_PROMPT = """You are an expert at analyzing technical documentation and source code
for a cloud service to identify its key domain topics.

Service: {hive_name} ({display_name})
Description: {description}

Below are sample document titles, headers, and code file names from this service's
knowledge base:

{content_sample}

Your job: Extract 15-25 concise routing topics that uniquely identify what this
service OWNS. These topics will be used by a router to decide which service should
answer a user's question.

RULES:
- Each topic should be 1-4 words (e.g., "backup policies", "cross-region restore")
- Focus on UNIQUE concepts this service owns — not generic terms like "API" or "error handling"
- Include both high-level features AND specific technical terms
- Include abbreviations/acronyms used by developers (e.g., "DPP", "CRR", "PITC")
- Order from most important to least important

Respond with ONLY a JSON array of topic strings, no other text:
["topic1", "topic2", "topic3", ...]
"""


class TopicExtractor:
    """Extracts routing topics from indexed hive content."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.store = DiscoveryStore()

    def extract_topics(
        self,
        hive_name: str,
        display_name: str,
        description: str,
        docs_collection_name: Optional[str] = None,
        code_collection_name: Optional[str] = None,
    ) -> List[str]:
        """Extract topics from a hive's indexed content.

        Samples content from ChromaDB collections and uses LLM to
        identify key domain topics.
        """
        content_sample = self._gather_content_sample(
            hive_name, docs_collection_name, code_collection_name
        )

        if not content_sample.strip():
            log.warning("No content found for hive '%s' — skipping topic extraction.", hive_name)
            return []

        prompt = TOPIC_EXTRACTION_PROMPT.format(
            hive_name=hive_name,
            display_name=display_name,
            description=description,
            content_sample=content_sample,
        )

        try:
            response = self.llm.generate(
                message=prompt,
                system="Extract domain topics. Return only a JSON array.",
                history=[],
            ).strip()

            topics = self._parse_topics(response)

            if topics:
                # Store in discovery DB
                self.store.set_topics(hive_name, topics, source="auto")
                log.info(
                    "Extracted %d topics for hive '%s': %s",
                    len(topics), hive_name, topics[:5],
                )
            return topics

        except Exception as e:
            log.error("Topic extraction failed for hive '%s': %s", hive_name, e)
            return []

    def refresh_all(self, hive_definitions: Dict) -> Dict[str, List[str]]:
        """Refresh topics for all hives.

        Parameters
        ----------
        hive_definitions : dict
            The hives.definitions section from config.yaml.

        Returns
        -------
        dict mapping hive_name → list of extracted topics.
        """
        results = {}
        for hive_name, hive_def in hive_definitions.items():
            display = hive_def.get("display_name", hive_name)
            desc = hive_def.get("description", "")
            docs_col = hive_def.get("vectorstore", {}).get("collection_name")
            code_col = hive_def.get("code_index", {}).get("collection_name")

            log.info("Extracting topics for hive '%s' (%s)...", hive_name, display)
            topics = self.extract_topics(
                hive_name=hive_name,
                display_name=display,
                description=desc,
                docs_collection_name=docs_col,
                code_collection_name=code_col,
            )
            results[hive_name] = topics

        return results

    def _gather_content_sample(
        self,
        hive_name: str,
        docs_collection: Optional[str],
        code_collection: Optional[str],
    ) -> str:
        """Gather a representative sample of content from the hive's collections."""
        import chromadb

        persist_dir = self.cfg.get("vectorstore", {}).get("persist_directory", ".chromadb")
        client = chromadb.PersistentClient(path=persist_dir)

        lines = []

        # Sample from docs collection
        if docs_collection:
            try:
                col = client.get_collection(docs_collection)
                count = col.count()
                if count > 0:
                    # Get a sample of documents with metadata
                    sample = col.peek(limit=min(count, 30))
                    lines.append("## Document Samples")

                    # Extract unique source files
                    sources = set()
                    for meta in (sample.get("metadatas") or []):
                        if meta and meta.get("source"):
                            sources.add(meta["source"])

                    if sources:
                        lines.append("Doc files:")
                        for s in sorted(sources):
                            lines.append(f"  - {Path(s).name}")

                    # Extract headers/content snippets
                    for doc in (sample.get("documents") or [])[:15]:
                        if doc:
                            # Get first line (usually a header)
                            first_line = doc.strip().split("\n")[0][:150]
                            lines.append(f"  Content: {first_line}")
            except Exception as e:
                log.debug("Could not sample docs collection '%s': %s", docs_collection, e)

        # Sample from code collection
        if code_collection:
            try:
                col = client.get_collection(code_collection)
                count = col.count()
                if count > 0:
                    sample = col.peek(limit=min(count, 30))
                    lines.append("\n## Code Samples")

                    # Extract unique file paths
                    code_files = set()
                    for meta in (sample.get("metadatas") or []):
                        if meta and meta.get("source"):
                            code_files.add(meta["source"])

                    if code_files:
                        lines.append("Code files:")
                        for f in sorted(code_files)[:30]:
                            lines.append(f"  - {Path(f).name}")

                    # Extract class/method names from code snippets
                    for doc in (sample.get("documents") or [])[:10]:
                        if doc:
                            # Look for class/method definitions
                            for match in re.finditer(
                                r'(class|public\s+\w+\s+\w+|interface)\s+(\w+)',
                                doc[:500]
                            ):
                                lines.append(f"  Symbol: {match.group(0)[:80]}")
            except Exception as e:
                log.debug("Could not sample code collection '%s': %s", code_collection, e)

        return "\n".join(lines)

    def _parse_topics(self, response: str) -> List[str]:
        """Parse LLM response into a list of topic strings."""
        import json

        # Try direct JSON parse first
        try:
            topics = json.loads(response)
            if isinstance(topics, list):
                return [t.strip().lower() for t in topics if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                topics = json.loads(match.group())
                if isinstance(topics, list):
                    return [t.strip().lower() for t in topics if isinstance(t, str) and t.strip()]
            except json.JSONDecodeError:
                pass

        # Fallback: parse line-by-line
        topics = []
        for line in response.splitlines():
            line = line.strip().strip("-•*").strip('"').strip("'").strip(",").strip()
            if line and len(line) < 60 and not line.startswith(("{", "[", "]", "}")):
                topics.append(line.lower())

        return topics[:25]
