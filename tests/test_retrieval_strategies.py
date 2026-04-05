"""Tests for the retrieval strategies — keyword, TF-IDF, semantic.

These tests use the mock evidence corpus directly; no API keys needed.
"""

from __future__ import annotations

import pytest

from src.retrieval.keyword import keyword_search
from src.retrieval.tfidf import tfidf_search
from src.retrieval.semantic import semantic_search
from src.retrieval.strategy_selector import select_strategy


_CORPUS = [
    {"text": "kubelet: node worker-3 condition DiskPressure=True", "domain": "node-metrics"},
    {"text": "Pod checkout-pod-7b evicted from node worker-3", "domain": "k8s-events"},
    {"text": "HTTP 503 — upstream connect error, service 'checkout' unreachable", "domain": "app-logs"},
    {"text": "OSPF adjacency on Gi0/0/0 Full, no changes detected", "domain": "network-logs"},
    {"text": "disk usage on /var/lib/kubelet reached 94%, eviction threshold is 90%", "domain": "node-metrics"},
    {"text": "Interface is up, line protocol is up", "domain": "network-logs"},
    {"text": "No BGP neighbor state changes in last 24 hours", "domain": "network-logs"},
    {"text": "Deployment 'checkout-app' Available replicas: 0", "domain": "k8s-events"},
]


class TestKeywordSearch:
    def test_finds_exact_matches(self):
        results = keyword_search("DiskPressure eviction", _CORPUS, top_k=3)
        texts = [r["text"] for r in results]
        assert any("DiskPressure" in t for t in texts)

    def test_returns_empty_for_no_match(self):
        results = keyword_search("xyzzy foobar", _CORPUS, top_k=3)
        assert results == []

    def test_respects_top_k(self):
        results = keyword_search("node worker checkout", _CORPUS, top_k=2)
        assert len(results) <= 2


class TestTfidfSearch:
    def test_ranks_relevant_higher(self):
        results = tfidf_search("disk pressure eviction kubelet", _CORPUS, top_k=3)
        assert len(results) > 0
        # The top result should mention disk or kubelet
        top_text = results[0]["text"].lower()
        assert "disk" in top_text or "kubelet" in top_text or "evict" in top_text

    def test_handles_empty_corpus(self):
        results = tfidf_search("anything", [], top_k=3)
        assert results == []


class TestSemanticSearch:
    def test_returns_results(self):
        # Uses TF-IDF fallback since no API key is set
        results = semantic_search("storage full pod removed", _CORPUS, top_k=3)
        assert len(results) > 0

    def test_handles_empty_corpus(self):
        results = semantic_search("anything", [], top_k=3)
        assert results == []


class TestStrategySelector:
    def test_selects_keyword_for_error_codes(self):
        h = {"text": "HTTP 503 error from checkout service", "domain": "application"}
        strategy = select_strategy(h, {"app-logs": _CORPUS})
        assert strategy == "keyword"

    def test_selects_tfidf_for_large_log_corpus(self):
        big_corpus = _CORPUS * 3  # >10 items in log domains
        h = {"text": "unusual pattern in application behavior", "domain": "application"}
        strategy = select_strategy(h, {"app-logs": big_corpus})
        assert strategy == "tfidf"

    def test_selects_semantic_for_infrastructure(self):
        h = {"text": "resource exhaustion on cluster nodes", "domain": "infrastructure"}
        strategy = select_strategy(h, {"k8s-events": _CORPUS[:2]})
        assert strategy == "semantic"

    def test_defaults_to_semantic(self):
        h = {"text": "something vague", "domain": "unknown"}
        strategy = select_strategy(h, {"misc": [{"text": "data"}]})
        assert strategy == "semantic"
