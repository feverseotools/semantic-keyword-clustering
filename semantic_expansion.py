"""Keyword expansion utilities using semantic relationships."""

from __future__ import annotations

import logging
from typing import Dict, List, Iterable

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except Exception:  # pragma: no cover - optional dependency
    api = None
    KeyedVectors = None

logger = logging.getLogger(__name__)


class SemanticExpansionEngine:
    """Suggest related keywords via word vectors and APIs."""

    def __init__(self, openai_client=None):
        self.client = openai_client
        self.word_vectors = self._load_word_vectors()

    def _load_word_vectors(self) -> KeyedVectors | None:
        if not api:
            logger.warning("gensim not available - vector expansion disabled")
            return None
        try:
            return api.load("glove-wiki-gigaword-50")
        except Exception as exc:  # pragma: no cover - might fail without internet
            logger.warning(f"Failed to load word vectors: {exc}")
            return None

    def _identify_core_concept(self, keywords: Iterable[str]) -> str:
        return next(iter(keywords), "")

    def _get_vector_similar(self, term: str) -> List[str]:
        if not self.word_vectors or term not in self.word_vectors:
            return []
        return [w for w, _s in self.word_vectors.most_similar(term, topn=5)]

    def _get_ai_related(self, term: str) -> List[str]:
        if not self.client:
            return []
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"List related keywords for {term}"}],
            )
            text = resp.choices[0].message.content
            return [t.strip() for t in text.split("\n") if t.strip()]
        except Exception as exc:  # pragma: no cover
            logger.warning(f"OpenAI expansion failed: {exc}")
            return []

    def _get_search_suggestions(self, term: str) -> List[str]:
        # Placeholder for search suggest API
        return []

    def _get_related_entities(self, term: str) -> List[str]:
        # Placeholder using wikidata
        return []

    def _rank_expansions(self, expansions: Dict[str, List[str]], existing: Iterable[str]) -> List[str]:
        seen = set(existing)
        results = []
        for method in expansions.values():
            for kw in method:
                if kw not in seen:
                    seen.add(kw)
                    results.append(kw)
        return results

    def expand_semantic_keywords(self, cluster_keywords: Dict[int, List[str]], expansion_factor: float = 2.0) -> Dict[int, List[str]]:
        expanded_keywords: Dict[int, List[str]] = {}
        for cid, keywords in cluster_keywords.items():
            core = self._identify_core_concept(keywords)
            expansions = {
                "word_vector_similar": self._get_vector_similar(core),
                "openai_semantic": self._get_ai_related(core),
                "search_suggest": self._get_search_suggestions(core),
                "related_entities": self._get_related_entities(core),
            }
            expanded_keywords[cid] = self._rank_expansions(expansions, keywords)
        return expanded_keywords

    def suggest_lsi_keywords(self, primary_keyword: str) -> List[str]:
        # Placeholder for LSI keyword generation
        return []
