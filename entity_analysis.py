"""Entity analysis utilities for semantic clustering."""

from __future__ import annotations

import logging
from typing import List, Dict, Any

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

import requests

logger = logging.getLogger(__name__)


def load_spacy_model_with_entities(model: str = "en_core_web_sm"):
    """Load a spaCy model for entity extraction."""
    if not spacy:
        logger.warning("spaCy not available - entity features disabled")
        return None
    try:
        return spacy.load(model)
    except Exception as exc:  # pragma: no cover - runtime protection
        logger.warning(f"Failed to load spaCy model {model}: {exc}")
        return None


class WikidataAPI:
    """Very small wrapper around the Wikidata search API."""

    endpoint = "https://www.wikidata.org/w/api.php"

    def search(self, query: str) -> List[Dict[str, Any]]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
        }
        try:
            r = requests.get(self.endpoint, params=params, timeout=5)
            if r.status_code == 200:
                return r.json().get("search", [])
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning(f"Wikidata search failed: {exc}")
        return []


class EntityAnalyzer:
    """Extract entities from keywords and build simple relationships."""

    def __init__(self, language: str = "English"):
        self.nlp = load_spacy_model_with_entities()
        self.knowledge_graph_api = WikidataAPI()
        self.language = language

    @staticmethod
    def _map_entity_type(label: str) -> str:
        mapping = {
            "PERSON": "persons",
            "ORG": "organizations",
            "GPE": "locations",
            "PRODUCT": "products",
        }
        return mapping.get(label, "topics")

    @staticmethod
    def _get_context_keywords(keyword: str, _ent) -> List[str]:
        return [keyword]

    def extract_entities(self, keywords_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from a list of keywords."""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "concepts": [],
            "products": [],
            "topics": [],
        }
        if not self.nlp:
            return entities

        for keyword in keywords_list:
            doc = self.nlp(keyword)
            for ent in doc.ents:
                key = self._map_entity_type(ent.label_)
                entities[key].append(
                    {
                        "text": ent.text,
                        "confidence": getattr(ent, "_.confidence", 1.0),
                        "context_keywords": self._get_context_keywords(keyword, ent),
                    }
                )
        return entities

    def _create_entity_graph(self, entities: Dict[str, List[Dict[str, Any]]]):
        try:
            import networkx as nx
        except Exception:  # pragma: no cover - optional dependency
            return None
        G = nx.Graph()
        for category, ents in entities.items():
            for ent in ents:
                G.add_node(ent["text"], category=category)
        return G

    def build_semantic_relationships(self, entities: Dict[str, List[Dict[str, Any]]]):
        """Map relationships between entities for topic clusters."""
        return self._create_entity_graph(entities)
