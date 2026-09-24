"""Candidate recall over the 29,860-code HTS index.

The plan's key design decision: never ask the model to choose from all ~30k codes.
Instead narrow first (cheap lexical/embedding recall), then let the Jev layer score
a small candidate set. Recall quality is the ceiling on final accuracy.

Uses BM25 (rank_bm25) over each entry's `search_text` (own description + parent
context). Deliberately simple and dependency-light: no GPU, no network, runs on
the Pi.
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA = Path(os.environ.get("DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
INDEX = DATA / "hts_index.json"

# Words that carry no discriminatory signal in HTS descriptions.
STOP = {
    "of", "the", "and", "or", "for", "with", "other", "than", "thereof", "not",
    "whether", "in", "to", "a", "an", "by", "from", "which", "are", "as", "on",
    "articles", "article", "including", "excluding", "such", "parts", "part",
    "nesoi", "no", "value", "any", "all", "unit", "units",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with light stemming (plural/participle folding)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = []
    for w in words:
        if w in STOP or len(w) < 2:
            continue
        # Cheap suffix folding so "screws"/"screw" and "plated"/"plate" match.
        for suf in ("ings", "ing", "ies", "ers", "er", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class HTSCatalog:
    def __init__(self, path: Path = INDEX):
        entries = json.loads(path.read_text())
        # Chapters frame the tree; they are not selectable classification codes.
        self.entries = [e for e in entries if e.get("level") != "chapter"]
        self.corpus = [tokenize(e["search_text"]) for e in self.entries]
        self.bm25 = BM25Okapi(self.corpus)

    def __len__(self) -> int:
        return len(self.entries)

    def recall(self, query: str, k: int = 40) -> list[dict]:
        """Return the top-k candidate entries by BM25 score."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # argpartition is O(n); only sort the top-k we actually return.
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            e = dict(self.entries[i])
            e["score"] = round(float(scores[i]), 4)
            out.append(e)
        return out

    def recall_filtered(self, query: str, headings: list[str] | None = None,
                        prefer_levels: tuple[str, ...] = ("subheading", "heading"),
                        k: int = 12) -> list[dict]:
        """Recall restricted to plausible headings, preferring meaningful levels.

        Why this exists: unrestricted BM25 over all ~30k codes surfaces
        lexically-similar noise (a query about metal fasteners ranked
        "buttons" highly because both mention "base metal"). And deep
        statistical leaves often read just "Other", which is useless as a
        candidate. So when the semantic filter has named headings, constrain to
        them and rank meaningful legal levels first.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)

        def collect(pred):
            out = []
            for i, e in enumerate(self.entries):
                if not pred(e):
                    continue
                if scores[i] <= 0:
                    continue
                out.append({**e, "score": round(float(scores[i]), 4)})
            out.sort(key=lambda x: -x["score"])
            return out

        head_set = set(headings or [])
        picked: list[dict] = []
        seen: set[str] = set()

        if head_set:
            # Pass 1: meaningful legal levels inside the named headings.
            for e in collect(lambda x: x["heading"] in head_set
                             and x["level"] in prefer_levels):
                if e["code"] not in seen:
                    picked.append(e)
                    seen.add(e["code"])
                if len(picked) >= k:
                    return picked
            # Pass 2: anything else inside the named headings (deeper leaves),
            # only to fill the remaining slots.
            for e in collect(lambda x: x["heading"] in head_set):
                if e["code"] not in seen:
                    picked.append(e)
                    seen.add(e["code"])
                if len(picked) >= k:
                    return picked

        # No headings (or not enough found): fall back to unrestricted recall,
        # still preferring meaningful levels.
        for e in collect(lambda x: x["level"] in prefer_levels):
            if e["code"] not in seen:
                picked.append(e)
                seen.add(e["code"])
            if len(picked) >= k:
                return picked
        for e in collect(lambda x: True):
            if e["code"] not in seen:
                picked.append(e)
                seen.add(e["code"])
            if len(picked) >= k:
                break
        return picked

    def get(self, code: str) -> dict | None:
        for e in self.entries:
            if e["code"] == code:
                return e
        return None


def _fmt(e: dict) -> str:
    """Render one candidate for the prompt: code + description (+ parent context)."""
    desc = e["description"]
    ctx = e.get("path_text") or ""
    if ctx and ctx.split(" > ")[-1] not in desc:
        return f"[{e['code']}] {desc} (under: {ctx})"
    return f"[{e['code']}] {desc}"


if __name__ == "__main__":
    cat = HTSCatalog()
    print(f"loaded {len(cat)} selectable codes")
    for q in ["zinc plated steel wood screws",
              "laptop computers portable",
              "assorted metal fasteners zinc plated mixed sizes"]:
        hits = cat.recall(q, k=6)
        print(f"\nquery: {q}")
        for h in hits:
            print(f"  {h['code']:>14}  {h['score']:>7}  {h['description'][:64]}")
