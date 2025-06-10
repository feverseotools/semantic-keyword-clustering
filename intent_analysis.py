"""Advanced search intent analysis using SERP data."""

from __future__ import annotations

import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class AdvancedIntentAnalyzer:
    """Analyze SERP patterns to refine search intent."""

    serp_api_endpoint = "https://serpapi.com/search.json"

    def fetch_serp_data(self, keyword: str, api_key: str) -> Dict[str, Any]:
        params = {"q": keyword, "api_key": api_key}
        try:
            r = requests.get(self.serp_api_endpoint, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"SERP API request failed: {exc}")
        return {}

    def _count_featured_snippets(self, serp_data: Dict[str, Any]) -> int:
        return int(bool(serp_data.get("featured_snippet")))

    def _count_comparison_pages(self, serp_data: Dict[str, Any]) -> int:
        return len(serp_data.get("comparison_results", []))

    def _count_product_pages(self, serp_data: Dict[str, Any]) -> int:
        return len(serp_data.get("shopping_results", []))

    def _count_brand_pages(self, serp_data: Dict[str, Any]) -> int:
        return len(serp_data.get("organic_results", []))

    def _calculate_refined_intent(self, signals: Dict[str, int], features: Dict[str, Any]) -> str:
        score = {
            "informational": signals.get("informational", 0),
            "commercial": signals.get("commercial", 0),
            "transactional": signals.get("transactional", 0),
            "navigational": signals.get("navigational", 0),
        }
        # Simple heuristic
        return max(score, key=score.get)

    def analyze_serp_patterns(self, keyword: str, serpapi_key: str) -> str:
        serp_data = self.fetch_serp_data(keyword, serpapi_key)
        intent_signals = {
            "informational": self._count_featured_snippets(serp_data),
            "commercial": self._count_comparison_pages(serp_data),
            "transactional": self._count_product_pages(serp_data),
            "navigational": self._count_brand_pages(serp_data),
        }
        features = {
            "featured_snippet": bool(serp_data.get("featured_snippet")),
            "people_also_ask": len(serp_data.get("people_also_ask", [])),
            "shopping_results": len(serp_data.get("shopping_results", [])),
            "local_pack": bool(serp_data.get("local_results")),
            "knowledge_panel": bool(serp_data.get("knowledge_graph")),
        }
        return self._calculate_refined_intent(intent_signals, features)
