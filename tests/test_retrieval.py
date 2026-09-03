"""Retrieval index tests."""

from __future__ import annotations

import json

import pytest

from procrastination_analyzer.retrieval import (
    Snippet,
    SnippetIndex,
    default_snippets_path,
    load_index,
    retrieve,
)


@pytest.fixture
def toy_index() -> SnippetIndex:
    return SnippetIndex(
        [
            Snippet(
                "a1",
                "avoidance",
                "Shrink the task",
                "Break the work into a tiny first step.",
                ("avoidance",),
            ),
            Snippet(
                "a2",
                "avoidance",
                "Shrink the task",
                "Pick the smallest slice that still counts.",
                ("avoidance",),
            ),
            Snippet(
                "f1",
                "fatigue",
                "Protect energy",
                "Move demanding work to your peak energy hours.",
                ("fatigue",),
            ),
            Snippet(
                "p1",
                "planning",
                "Define done",
                "Write what finished looks like before starting.",
                ("planning",),
            ),
        ]
    )


class TestIndexConstruction:
    def test_rejects_empty_corpus(self):
        with pytest.raises(ValueError, match="empty snippet corpus"):
            SnippetIndex([])

    def test_packaged_corpus_loads(self):
        index = load_index()
        assert len(index) > 50
        assert "avoidance" in index.categories

    def test_packaged_corpus_has_no_duplicate_ids(self):
        raw = json.loads(default_snippets_path().read_text(encoding="utf-8"))
        ids = [s["id"] for s in raw]
        assert len(ids) == len(set(ids))

    def test_packaged_corpus_has_no_duplicate_texts(self):
        """The original corpus held 220 entries but only 122 distinct texts."""
        raw = json.loads(default_snippets_path().read_text(encoding="utf-8"))
        texts = [" ".join(s["text"].lower().split()) for s in raw]
        assert len(texts) == len(set(texts))

    def test_path_resolves_independently_of_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert default_snippets_path().exists()
        assert len(load_index()) > 0


class TestSearch:
    def test_returns_requested_count(self, toy_index):
        assert len(toy_index.search("break the task down", k=2)) == 2

    def test_results_are_sorted_by_score(self, toy_index):
        scores = [r["score"] for r in toy_index.search("energy hours", k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_never_returns_duplicate_snippets(self, toy_index):
        results = toy_index.search("task", k=4)
        assert len({r["id"] for r in results}) == len(results)

    def test_diversify_prefers_distinct_titles(self, toy_index):
        results = toy_index.search("smallest slice tiny step", k=2, diversify=True)
        assert len({r["title"] for r in results}) == 2

    def test_category_filter_restricts_results(self, toy_index):
        results = toy_index.search("anything", k=4, category="fatigue")
        assert {r["category"] for r in results} == {"fatigue"}

    def test_unknown_category_falls_back_to_full_corpus(self, toy_index):
        assert toy_index.search("task", k=2, category="nonexistent")

    def test_min_score_filters_weak_matches(self, toy_index):
        assert toy_index.search("xylophone quantum", k=4, min_score=0.2) == []

    def test_empty_query_rejected(self, toy_index):
        with pytest.raises(ValueError, match="non-empty"):
            toy_index.search("   ")

    def test_k_larger_than_corpus_is_safe(self, toy_index):
        results = toy_index.search("task", k=99)
        assert len(results) <= len(toy_index)

    def test_query_terms_do_not_pollute_the_index(self, toy_index):
        """The original fit the vectorizer on corpus+query, leaking query IDF."""
        first = toy_index.search("energy", k=3)
        toy_index.search("completely unrelated zebra vocabulary", k=3)
        second = toy_index.search("energy", k=3)
        assert [r["score"] for r in first] == [r["score"] for r in second]


class TestCaching:
    def test_load_index_is_cached(self):
        assert load_index() is load_index()

    def test_module_level_retrieve_works(self):
        results = retrieve("avoidance long gaps", k=3, category="avoidance")
        assert len(results) == 3
        assert all("score" in r for r in results)
