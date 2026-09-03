"""TF-IDF retrieval over the suggestion corpus.

Two problems with the original implementation are fixed here:

* It reloaded the JSON corpus and refit the ``TfidfVectorizer`` on *every*
  call, so vocabulary and IDF weights were rebuilt from scratch each time a user
  moved a slider. The index is now built once and cached.
* It appended the query to the corpus before fitting, which leaks query terms
  into the IDF statistics and makes scores depend on what was asked. The
  vectorizer is now fit on the corpus alone and the query is only transformed.

Retrieval is intentionally lexical rather than embedding-based: the corpus is
~120 short snippets, so a dense model would add a heavy dependency and offer
little over TF-IDF at this scale. :func:`SnippetIndex.search` documents the
trade-off.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["Snippet", "SnippetIndex", "default_snippets_path", "load_index", "retrieve"]


def default_snippets_path() -> Path:
    """Path to the packaged snippet corpus.

    Resolved relative to this module so it works regardless of the working
    directory — the original code hard-coded ``"procrastination_analyzer/rag/
    snippets.json"``, which only resolved when launched from the repo root.
    """
    return Path(__file__).resolve().parent / "data" / "snippets.json"


@dataclass(frozen=True)
class Snippet:
    """One retrievable suggestion."""

    id: str
    category: str
    title: str
    text: str
    tags: tuple[str, ...] = ()

    def to_dict(self, score: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "tags": list(self.tags),
        }
        if score is not None:
            payload["score"] = round(float(score), 4)
        return payload


class SnippetIndex:
    """An in-memory TF-IDF index over the suggestion corpus."""

    def __init__(self, snippets: Sequence[Snippet]) -> None:
        if not snippets:
            raise ValueError("Cannot build an index over an empty snippet corpus.")
        self.snippets: tuple[Snippet, ...] = tuple(snippets)

        # Indexing title + tags alongside body text meaningfully improves recall
        # on short queries, since the body often omits the topic word entirely.
        corpus = [f"{s.title} {' '.join(s.tags)} {s.text}" for s in self.snippets]

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def __len__(self) -> int:
        return len(self.snippets)

    @property
    def categories(self) -> list[str]:
        """Sorted list of distinct categories in the corpus."""
        return sorted({s.category for s in self.snippets})

    @classmethod
    def from_json(cls, path: Path | None = None) -> SnippetIndex:
        """Load a corpus from a JSON file and build the index."""
        path = Path(path) if path is not None else default_snippets_path()
        if not path.exists():
            raise FileNotFoundError(f"Snippet corpus not found at {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        snippets = [
            Snippet(
                id=str(item.get("id", f"s{i}")),
                category=str(item.get("category", "general")),
                title=str(item.get("title", "Suggestion")),
                text=str(item["text"]),
                tags=tuple(item.get("tags", ())),
            )
            for i, item in enumerate(raw)
        ]
        return cls(snippets)

    def search(
        self,
        query: str,
        *,
        k: int = 4,
        category: str | None = None,
        min_score: float = 0.0,
        diversify: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the ``k`` best-matching snippets for ``query``.

        Args:
            query: Free-text query, typically built from the detected pattern.
            k: Maximum results.
            category: Restrict to one category. Ignored if it would empty the
                candidate pool.
            min_score: Drop results below this cosine similarity. Returning a
                near-zero match is worse than returning fewer suggestions.
            diversify: Avoid returning several snippets with the same title.
                The corpus reuses titles across variants, so the naive top-k is
                often four rewordings of one idea.

        Returns:
            Snippet dicts with an added ``score`` key, best first.
        """
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        candidate_idx = np.arange(len(self.snippets))
        if category:
            mask = np.array([s.category == category for s in self.snippets])
            if mask.any():
                candidate_idx = candidate_idx[mask]

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix[candidate_idx]).ravel()

        order = [int(i) for i in np.argsort(-sims) if float(sims[i]) >= min_score]
        if not diversify:
            return [
                self.snippets[int(candidate_idx[pos])].to_dict(float(sims[pos]))
                for pos in order[:k]
            ]

        # Progressively relax the per-title cap rather than topping up with the
        # snippets diversification just rejected: raising the cap keeps results
        # in score order and can never emit the same snippet twice.
        max_title_cap = max(1, k)
        for cap in range(1, max_title_cap + 1):
            results: list[dict[str, Any]] = []
            title_counts: dict[str, int] = {}
            for pos in order:
                snippet = self.snippets[int(candidate_idx[pos])]
                if title_counts.get(snippet.title, 0) >= cap:
                    continue
                title_counts[snippet.title] = title_counts.get(snippet.title, 0) + 1
                results.append(snippet.to_dict(float(sims[pos])))
                if len(results) >= k:
                    return results
            # Exhausted the candidate pool at this cap; if raising it cannot add
            # anything (we already saw every candidate), stop here.
            if len(results) == len(order):
                return results
        return results


@lru_cache(maxsize=4)
def load_index(path: str | None = None) -> SnippetIndex:
    """Load and cache the snippet index.

    Cached on path, so repeated calls (every Streamlit rerun, every CLI
    invocation in a loop) reuse the fitted vectorizer instead of refitting it.
    """
    return SnippetIndex.from_json(Path(path) if path else None)


def retrieve(
    query: str,
    *,
    k: int = 4,
    category: str | None = None,
    snippets_path: str | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Convenience wrapper around the cached index."""
    return load_index(snippets_path).search(query, k=k, category=category, min_score=min_score)
