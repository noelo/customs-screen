"""Three-tier customs classifier (the "Border Classifier" demo).

Tier 1  - RECALL:        BM25 over the 29,860-code HTS index -> candidate pool.
Tier 1b - SEMANTIC FILTER: one cheap LLM call maps the goods description to the
          plausible HTS headings, then the pool is intersected/expanded. This is
          the step that rescues lexical recall's blind spot (e.g. "laptop" never
          appears in HTS; the text says "portable automatic data processing
          machines"). It reasons, so it uses the CHAT endpoint.
Tier 2  - JEV SCORE:     the raw /v1/completions logprob read over single-token
          labels -> restricted-softmax distribution over the candidates. THIS is
          the decision, and it is the cheap layer.
Tier 3  - VERIFY/ESCALATE: an LLM verifies the winner against the goods narrative
          and explains it; a flat distribution escalates to a human broker.

Design note: the reason Tier 2 must be logged is the whole point of the demo --
the *decision* is a single forward pass over ~30 candidates, while the frontier
model is spent only on verification and only when the spread says so.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from jev import MAX_LABELS_PER_CALL, JevScorer, ScoringError, _load_key
from recall import HTSCatalog

DEFAULT_BASE_URL = "https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/glm-53-flash/v1"
DEFAULT_MODEL = "glm-53-flash"


@dataclass
class Classification:
    query: str
    code: str | None
    description: str | None
    confidence: float
    margin: float
    distribution: dict[str, float]
    candidates: list[dict]
    tier_used: int
    escalated: bool
    rationale: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    filter_headings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _chat(prompt: str, base_url: str, model: str, max_tokens: int = 800,
          timeout: int = 90) -> str:
    """One chat completion (used for the filter and the verifier, never the decision)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _load_key()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
            msg = d["choices"][0]["message"]
            return (msg.get("content") or msg.get("reasoning") or "").strip()
    except urllib.error.HTTPError as e:
        raise ScoringError(f"chat request failed: HTTP {e.code} {e.read().decode()[:200]}") from e
    except Exception as e:
        raise ScoringError(f"chat request failed: {type(e).__name__}: {e}") from e


class CustomsClassifier:
    def __init__(self, catalog: HTSCatalog | None = None,
                 scorer: JevScorer | None = None,
                 base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 pool_size: int = 30, use_semantic_filter: bool = True,
                 use_verifier: bool = True):
        self.catalog = catalog or HTSCatalog()
        self.scorer = scorer or JevScorer(base_url=base_url, model=model)
        self.base_url = base_url
        self.model = model
        self.pool_size = pool_size
        self.use_semantic_filter = use_semantic_filter
        self.use_verifier = use_verifier

    # ---- Tier 1b: semantic heading filter -----------------------------------
    def _filter_headings(self, query: str) -> list[str]:
        """Ask the model which HTS headings plausibly apply. Reasoning is fine here."""
        prompt = (
            "You are a customs classification assistant.\n"
            "Given a description of imported goods, list the 3-6 most plausible "
            "4-digit Harmonized Tariff Schedule (HTS) headings.\n"
            "Headings are 4 digits, e.g. 7318 = screws/bolts/nuts of iron or steel, "
            "8471 = automatic data processing machines, 8517 = telephones.\n\n"
            f"Goods: {query}\n\n"
            "Reply with ONLY comma-separated 4-digit headings, most likely first. "
            "No prose."
        )
        out = _chat(prompt, self.base_url, self.model, max_tokens=300)
        return re.findall(r"\b(\d{4})\b", out)[:6]

    # ---- Tier 1: candidate pool --------------------------------------------
    def _build_pool(self, query: str, headings: list[str]) -> tuple[list[dict], list[str]]:
        notes: list[str] = []
        # When the semantic filter named headings, constrain recall to them and
        # prefer meaningful legal levels (subheading/heading) over deep
        # statistical leaves that read just "Other".
        pool = self.catalog.recall_filtered(
            query, headings=headings or None, k=self.pool_size)
        if headings:
            notes.append(f"semantic filter constrained recall to headings {headings}")
        if not pool:
            pool = self.catalog.recall(query, k=self.pool_size)
            notes.append("recall restricted to headings found nothing; used unrestricted recall")
        return pool, notes

    # ---- Tier 2: Jev scoring ------------------------------------------------
    def _label_for(self, i: int) -> str:
        return chr(ord("A") + i)

    @staticmethod
    def _render_candidate(e: dict) -> str:
        """Render one HTS entry as a candidate line for the scoring prompt."""
        text = f"[{e['code']}] {e['description']}"
        if e.get("path_text"):
            text += f" (under: {e['path_text']})"
        return text

    def _score_bracketed_with_set(self, query: str,
                                  pool: list[dict]) -> tuple[object, list[dict]]:
        """Run the bracket tournament and return (decision, survivor_entries).

        The returned decision's label alphabet indexes `survivor_entries`, so the
        caller can resolve winners and the top-3 without guessing which subset
        the final round actually scored.
        """
        groups = [pool[i:i + MAX_LABELS_PER_CALL]
                  for i in range(0, len(pool), MAX_LABELS_PER_CALL)]
        survivors: list[dict] = []
        for group in groups:
            strs = [self._render_candidate(e) for e in group]
            d = self.scorer.score(query, strs)
            survivors.append(group[d.index])
        final_strs = [self._render_candidate(e) for e in survivors]
        decision = self.scorer.score(query, final_strs)
        return decision, survivors

    def classify(self, query: str) -> Classification:
        timings: dict[str, float] = {}
        notes: list[str] = []

        # Tier 1b -- cheap semantic filter (reasons, so it uses the chat endpoint).
        import time
        headings: list[str] = []
        if self.use_semantic_filter:
            t = time.perf_counter()
            try:
                headings = self._filter_headings(query)
            except ScoringError as e:
                notes.append(f"semantic filter failed, falling back to lexical only: {e}")
            timings["filter_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # Tier 1 -- candidate recall.
        t = time.perf_counter()
        pool, pool_notes = self._build_pool(query, headings)
        notes.extend(pool_notes)
        timings["recall_ms"] = round((time.perf_counter() - t) * 1000, 1)

        if not pool:
            return Classification(query=query, code=None, description=None,
                                  confidence=0.0, margin=0.0, distribution={},
                                  candidates=[], tier_used=1, escalated=True,
                                  rationale="No candidates found by recall.",
                                  timings_ms=timings, notes=notes)

        # Tier 2 -- the decision. `decision_set` is the exact list the final
        # label alphabet indexed, so labels/indices always resolve correctly
        # whether or not the bracket tournament ran.
        candidate_strs = [self._render_candidate(e) for e in pool]
        t = time.perf_counter()
        if len(candidate_strs) > MAX_LABELS_PER_CALL:
            decision, decision_set = self._score_bracketed_with_set(query, pool)
            notes.append(
                f"pool of {len(candidate_strs)} exceeded the {MAX_LABELS_PER_CALL}-"
                f"label logprob window; bracketed tournament scored "
                f"{-(-len(candidate_strs) // MAX_LABELS_PER_CALL)} groups, then a "
                f"final round over {len(decision_set)} survivors"
            )
        else:
            decision_set = pool
            labels = [self._label_for(i) for i in range(len(candidate_strs))]
            decision = self.scorer.score(query, candidate_strs, labels)
        timings["jev_ms"] = round((time.perf_counter() - t) * 1000, 1)

        winner = decision_set[decision.index]
        display_labels = [self._label_for(i) for i in range(len(pool))]
        result = Classification(
            query=query,
            code=winner["code"],
            description=winner["description"],
            confidence=decision.confidence,
            margin=decision.margin,
            distribution=decision.probabilities,
            candidates=[{"label": display_labels[i], "code": pool[i]["code"],
                         "description": pool[i]["description"]}
                        for i in range(len(pool))],
            tier_used=2,
            escalated=decision.escalate,
            timings_ms=timings,
            filter_headings=headings,
            notes=notes,
        )

        # Tier 3 -- spend frontier tokens only when the spread says so.
        if decision.escalate and self.use_verifier:
            t = time.perf_counter()
            top_sorted = sorted(decision.probabilities.items(), key=lambda kv: -kv[1])[:3]
            lines = []
            for lab, _ in top_sorted:
                # Labels are positional (A, B, C, ...) over decision_set in both
                # scoring modes, so the letter maps straight to an index.
                idx = ord(lab) - ord("A")
                ent = decision_set[idx] if 0 <= idx < len(decision_set) else winner
                lines.append(f"{lab}. [{ent['code']}] {ent['description']}")
            top3_text = "\n".join(lines)
            prompt = (
                "You are a customs broker verifying an automated classification.\n"
                f"Goods: {query}\n\n"
                "The automated system is uncertain between these candidates:\n"
                f"{top3_text}\n\n"
                "Which single HTS code is correct, and why? State the code, then one "
                "short paragraph of reasoning citing the goods' material and function."
            )
            try:
                result.rationale = _chat(prompt, self.base_url, self.model)
                result.tier_used = 3
            except ScoringError as e:
                result.notes.append(f"verifier failed: {e}")
            timings["verify_ms"] = round((time.perf_counter() - t) * 1000, 1)

        return result
