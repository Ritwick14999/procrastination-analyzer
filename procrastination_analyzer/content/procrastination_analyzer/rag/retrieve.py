
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@dataclass
class Snippet:
    id: str
    category: str
    title: str
    text: str
    tags: List[str]

def load_snippets(path: str) -> List[Snippet]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Snippet] = []
    for s in raw:
        out.append(
            Snippet(
                id=s["id"],
                category=s.get("category", "general"),
                title=s.get("title", "Suggestion"),
                text=s["text"],
                tags=s.get("tags", []),
            )
        )
    return out

def retrieve_snippets(query: str, snippets_path: str, k: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
    snippets = load_snippets(snippets_path)
    if category:
        filtered = [s for s in snippets if s.category == category]
        if filtered:
            snippets = filtered

    corpus = [s.text for s in snippets]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus + [query])

    sims = linear_kernel(tfidf[-1], tfidf[:-1]).flatten()
    top_idx = sims.argsort()[-k:][::-1]

    results: List[Dict[str, Any]] = []
    for i in top_idx:
        s = snippets[int(i)]
        results.append(
            {
                "id": s.id,
                "category": s.category,
                "title": s.title,
                "text": s.text,
                "tags": s.tags,
                "score": round(float(sims[int(i)]), 4),
            }
        )
    return results
